"""Inference-only ONSET difficulty model (trained by onset_colab_train_806_grid_2.py).

A score is one or more MusicXML files, each given with its start tempo (quarter-note bpm). Every file
is encoded into an event matrix (one row per onset: 88 + 88 key bits, 12 + 12 attack-rate bits, 12
timing values), exactly as in training (encoder version 3, rate from the inter-onset interval). The
matrices of a multi-file score are concatenated in file order. Each fold model predicts a grade 1..9;
the score's grade is their mean, rounded half up.

    model = DifficultyModel("checkpoints")
    model.predict([("sonata-1.musicxml", 120.0), ("sonata-2.musicxml", 66.0)])
    -> {"grade": 6, "average": 5.8, "folds": [6, 6, 5, 6, 6]}
"""
import bisect
import glob
import math
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from music21 import converter, meter, tempo
from torch import nn

LO, HI = 21, 108                       # MIDI range of the 88 keys
BLACK = {1, 3, 6, 8, 10}               # pitch classes of black keys
RATE_EDGES = [30, 45, 70, 100, 150, 220, 330, 500, 750, 1100, 1600]   # attacks per minute, 12 bins
SANITIZE_TAGS = ("harmony", "arpeggiate", "non-arpeggiate")
UNIT_IN_QUARTERS = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "16th": 0.25, "32nd": 0.125}


# ---------------------------------------------------------------------------------------------
# Start tempo
# ---------------------------------------------------------------------------------------------
def metronome_bpm(m):
    """A printed <metronome> as quarter-note bpm, None when it has no number (e.g. quarter = half)."""
    units = m.findall("beat-unit")
    try:
        bpm = float(m.findtext("per-minute"))
    except (TypeError, ValueError):
        return None
    unit = units[0].text.strip() if len(units) == 1 and units[0].text else None
    if unit not in UNIT_IN_QUARTERS or bpm <= 0:
        return None
    return bpm * UNIT_IN_QUARTERS[unit] * (2 - 0.5 ** len(m.findall("beat-unit-dot")))


def start_tempo(path):
    """Quarter-note bpm of a printed numeric metronome mark at the very start of the file (time 0 of the
    first measure), else None. A mark later in the first measure does not count."""
    root = ET.parse(path).getroot()
    for part in root.findall("part"):
        first = part.find("measure")
        t = 0                                               # time in divisions from the start of the measure
        for el in ([] if first is None else first):
            if el.tag == "note" and el.find("chord") is None:
                t += int(el.findtext("duration") or 0)     # grace notes have no duration
            elif el.tag in ("backup", "forward"):
                t += int(el.findtext("duration")) * (-1 if el.tag == "backup" else 1)
            elif el.tag == "direction" and t + int(el.findtext("offset") or 0) == 0:
                for m in el.iter("metronome"):
                    if metronome_bpm(m):
                        return metronome_bpm(m)
    return None


# ---------------------------------------------------------------------------------------------
# Encoder: MusicXML file -> event matrix (T, 212)
# ---------------------------------------------------------------------------------------------
def _metronome_direction(bpm):
    d = ET.Element("direction")
    m = ET.SubElement(ET.SubElement(d, "direction-type"), "metronome")
    ET.SubElement(m, "beat-unit").text = "quarter"
    ET.SubElement(m, "per-minute").text = bpm
    return d


def _sanitize(root):
    """As in training: drop chord symbols / arpeggio brackets and display steps that are not A..G,
    and turn playback-only <sound tempo> into explicit quarter-note metronome marks."""
    for parent in root.iter():
        for child in list(parent):
            if child.tag in SANITIZE_TAGS:
                parent.remove(child)
    for el in root.iter():
        step = el.find("display-step") if el.tag in ("rest", "unpitched") else None
        if step is not None and (step.text or "").strip().upper() not in list("ABCDEFG"):
            for child in list(el):
                if child.tag in ("display-step", "display-octave"):
                    el.remove(child)
    for measure in root.iter("measure"):
        for el in list(measure):
            if el.tag == "direction":
                sound = el.find("sound")
                if sound is None or not sound.get("tempo"):
                    continue
                if el.find("direction-type/metronome") is None:
                    after = [j for j, c in enumerate(list(el)) if c.tag == "direction-type"]
                    el.insert((after[-1] + 1) if after else 0, _metronome_direction(sound.get("tempo"))[0])
                del sound.attrib["tempo"]
            elif el.tag == "sound" and el.get("tempo"):
                measure.insert(list(measure).index(el), _metronome_direction(el.get("tempo")))
                del el.attrib["tempo"]


def _parse(path):
    tree = ET.parse(path)
    _sanitize(tree.getroot())
    fd, tmp = tempfile.mkstemp(suffix=".musicxml")
    os.close(fd)
    try:
        tree.write(tmp, encoding="UTF-8", xml_declaration=True)
        return converter.parse(tmp)
    finally:
        os.remove(tmp)


def _rate_onehot(rate):
    v = [0.0] * (len(RATE_EDGES) + 1)
    if rate is not None:
        v[sum(1 for e in RATE_EDGES if rate >= e)] = 1.0
    return v


def encode_file(path, start_bpm):
    """One MusicXML file -> float32 array (onsets, 212). The start tempo holds from offset 0; later
    tempo marks of the file are honoured. Hands = the first two staves with notes."""
    s = _parse(path)
    parts = [p for p in s.parts if len(p.recurse().notes) > 0]
    if len(parts) < 2:
        raise ValueError(f"expected two staves with notes (right hand / left hand), found {len(parts)}")
    rh_part, lh_part = parts[0], parts[1]

    # tempo map: offset in quarters -> quarter-note bpm
    Q = lambda x: round(float(x), 4)  # noqa: E731
    tm = {}
    for mm in s.recurse().getElementsByClass(tempo.MetronomeMark):
        if mm.getQuarterBPM():
            tm[Q(mm.getOffsetInHierarchy(s))] = float(mm.getQuarterBPM())
    marks = [(0.0, float(start_bpm))] + [(o, b) for o, b in sorted(tm.items()) if o > 0]

    def sec_at(t):
        total = 0.0
        for (o, b), (o2, _) in zip(marks, marks[1:] + [(None, None)]):
            end = t if o2 is None or o2 > t else o2
            if end > o:
                total += float(end - o) * 60 / b
            if o2 is None or o2 >= t:
                break
        return total

    # bar starts and time signatures; a bar split across a system break counts as one bar
    bar_starts, ts_map = [], {}
    prev, bar_len = None, None
    for m in rh_part.getElementsByClass("Measure"):
        off = Q(m.offset)
        ts = m.getElementsByClass(meter.TimeSignature)
        if ts:
            bar_len = float(ts[0].barDuration.quarterLength)
            ts_map[off] = (ts[0].numerator, ts[0].denominator)
        cont = (prev is not None and bar_len is not None
                and any(b.type == "none" for b in prev.getElementsByClass("Barline")
                        if getattr(b, "location", "right") in ("right", None))
                and float(prev.duration.quarterLength) < bar_len - 1e-6
                and float(prev.duration.quarterLength) + float(m.duration.quarterLength) <= bar_len + 1e-6)
        if not cont:
            bar_starts.append(off)
        prev = m
    bar_starts.sort()

    def bar_info(t):
        b = [k for k in bar_starts if k <= t][-1]
        ts = [ts_map[k] for k in sorted(ts_map) if k <= b]
        return t - b, (ts[-1] if ts else (4, 4))

    def events(part):
        d = {}
        for x in part.stripTies(matchByPitch=False).flatten().notes:
            if float(x.duration.quarterLength) == 0:          # grace notes
                continue
            d.setdefault(Q(x.offset), set()).update(p.midi for p in x.pitches if LO <= p.midi <= HI)
        return {k: v for k, v in d.items() if v}

    rh, lh = events(rh_part), events(lh_part)
    if not rh and not lh:
        raise ValueError("no notes in either staff")
    rh_keys, lh_keys = sorted(rh), sorted(lh)

    def ioi_next(keys, t, t_sec):
        i = bisect.bisect_right(keys, t)
        return sec_at(keys[i]) - t_sec if i < len(keys) else None

    keys = lambda notes: [1.0 if p in notes else 0.0 for p in range(LO, HI + 1)]  # noqa: E731
    log_ioi = lambda x: math.log1p(round(x, 4)) if x else 0.0                     # noqa: E731
    rows, prev_t = [], None
    for t in sorted(set(rh) | set(lh)):
        r, l = rh.get(t, set()), lh.get(t, set())
        t_sec = sec_at(t)
        r_ioi = ioi_next(rh_keys, t, t_sec) if r else None
        l_ioi = ioi_next(lh_keys, t, t_sec) if l else None
        beat, (tsn, tsd) = bar_info(t)
        rows.append(keys(r) + keys(l)
                    + _rate_onehot(60 / r_ioi if r_ioi else None) + _rate_onehot(60 / l_ioi if l_ioi else None)
                    + [log_ioi(t_sec - prev_t) if prev_t is not None else 0.0, log_ioi(r_ioi), log_ioi(l_ioi),
                       len(r) / 4, len(l) / 4,
                       (max(r) - min(r)) / 12 if r else 0.0, (max(l) - min(l)) / 12 if l else 0.0,
                       sum(p % 12 in BLACK for p in r) / 4, sum(p % 12 in BLACK for p in l) / 4,
                       float(bool(r) and bool(l)), float(abs(beat) < 1e-6),
                       min(round(float(beat), 3) / (tsn * 4.0 / tsd), 1.0)])
        prev_t = t_sec
    return np.array(rows, dtype=np.float32)


def encode_score(files):
    """[(path, start_bpm), ...] -> the files' event matrices concatenated in order."""
    return np.concatenate([encode_file(path, bpm) for path, bpm in files])


# ---------------------------------------------------------------------------------------------
# Network: GRU -> context attention -> ordinal head (parameter names as in training)
# ---------------------------------------------------------------------------------------------
class ContextAttention(nn.Module):
    def __init__(self, size, num_head=1):
        super().__init__()
        self.size, self.num_head, self.head_size = size, num_head, size // num_head
        self.attention_net = nn.Linear(size, size)
        self.context_vector = nn.Parameter(torch.zeros(num_head, self.head_size, 1))

    def forward(self, x):                                   # (B, T, size) -> (B, size)
        B, T, _ = x.shape
        att = torch.tanh(self.attention_net(x)).view(B, T, self.num_head, self.head_size)
        weights = torch.softmax(torch.einsum("bthd,hdo->bth", att, self.context_vector), dim=1)
        heads = x.view(B, T, self.num_head, self.head_size)
        return torch.einsum("bth,bthd->bhd", weights, heads).reshape(B, self.size)


class OnsetNet(nn.Module):
    def __init__(self, input_size, hidden, num_layers, num_head, bidirectional, n_classes):
        super().__init__()
        self.branch = nn.ModuleDict({"gru": nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True,
                                                   bidirectional=bidirectional)})
        size = hidden * (2 if bidirectional else 1)
        self.attn = ContextAttention(size, num_head)
        self.fc = nn.Linear(size, n_classes)

    def forward(self, x):                                   # (1, T, D) -> (1, n_classes)
        out, _ = self.branch["gru"](x)
        return self.fc(torch.relu(self.attn(out)))


class DifficultyModel:
    """The fold ensemble in `ckpt_dir` (onset_ordinal-<fold>.pt)."""

    def __init__(self, ckpt_dir, device="cpu"):
        self.device, self.folds = device, []
        for path in sorted(glob.glob(os.path.join(ckpt_dir, "onset_ordinal-*.pt"))):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck.get("encoder_version") != 3 or ck.get("rate_from", "ioi") != "ioi":
                raise ValueError(f"{path}: needs encoder version 3 with rates from the IOI")
            net = OnsetNet(ck["input_size"], ck["hidden"], ck["num_layers"], ck["num_head"],
                           ck["bidirectional"], ck["n_classes"])
            net.load_state_dict(ck["state_dict"])
            self.folds.append((net.to(device).eval(), ck["input_size"], ck.get("max_len", 0)))
        if not self.folds:
            raise FileNotFoundError(f"no onset_ordinal-*.pt in {ckpt_dir}")

    @torch.no_grad()
    def predict_matrix(self, x):
        """Event matrix (T, 212) -> {"grade", "average", "folds"}. A fold's 'below grade 1' counts as 1."""
        grades = []
        for net, input_size, max_len in self.folds:
            xi = torch.from_numpy(x[-max_len:] if max_len else x)[:, :input_size]   # core = the first 200 columns
            logits = net(xi.unsqueeze(0).to(self.device))
            grades.append(max(int((logits > 0.5).cumprod(dim=1).sum()) - 1, 0) + 1)
        average = sum(grades) / len(grades)
        return {"grade": int(math.floor(average + 0.5)), "average": round(average, 3), "folds": grades}

    def predict(self, files):
        """[(musicxml_path, start_bpm), ...] (one entry per file of the score) -> {"grade", "average", "folds"}."""
        return self.predict_matrix(encode_score(files))
