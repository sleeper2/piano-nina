"""Parse every score in ../score_musicxml once and write per-score content statistics and header metadata.

Outputs (results/):
  content.csv   one row per score: length, written duration at the stored tempo, pitch, polyphony, voices,
                chord span, shortest note, time/key signatures, and counts of notational markings
  metadata.csv  one row per score: header text (work/movement title, creators, rights, credits, part names),
                used to classify arrangements, genre and form
  key_counts.csv one row per score: note attacks on each of the 88 piano keys (columns "21" .. "108", MIDI
                numbers; tied continuations and grace notes left out) and "out_of_range" for any other pitch

Durations are the written length (repeats and D.C./D.S. are not unfolded). The tempo is the playback tempo
stored in <sound tempo> (quarter notes per minute); a score with none plays at MuseScore's default 120.
"""
import os
import sys
from collections import defaultdict
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import pandas as pd
from lxml import etree

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORES = os.environ.get("SCORES", os.path.join(ROOT, "..", "score_musicxml"))
OUT = os.path.join(ROOT, "results")

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
TYPE_ORDER = ["maxima", "long", "breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th",
              "256th", "512th", "1024th"]
DEFAULT_TEMPO = 120.0


def text(el, path):
    found = el.find(path)
    return found.text.strip() if found is not None and found.text else ""


def metadata(root):
    creators = "; ".join(f"{c.get('type', '')}: {' '.join((c.text or '').split())}" for c in root.iter("creator"))
    credits = " / ".join(" ".join("".join(c.itertext()).split()) for c in root.iter("credit-words")
                         if "".join(c.itertext()).strip())
    parts = "; ".join(" ".join((p.findtext("part-name") or "").split()) for p in root.iter("score-part"))
    return {
        "work_title": " ".join(text(root, "work/work-title").split()),
        "movement_title": " ".join(text(root, "movement-title").split()),
        "creators": creators,
        "rights": " ".join(" / ".join(r.text or "" for r in root.iter("rights")).split()),
        "credits": credits[:400],
        "part_names": parts,
        "software": text(root, "identification/encoding/software"),
    }


def parse(path, exact=False):
    """Walk the score once. With exact=True onsets and durations are Fractions of a quarter note (tuplets stay
    exact); otherwise floats."""
    num = (lambda x: Fraction(str(x).strip())) if exact else float
    root = etree.parse(path).getroot()
    notes = []                      # (part, staff, voice, onset_q, dur_q, midi, attack)
    note_types = []
    grace = 0
    tempos = []                     # (onset_q, qpm)
    measure_len = defaultdict(lambda: num(0))
    measure_voices = defaultdict(set)   # (part, staff, measure) -> voices with notes
    time_sigs, key_sigs = [], []
    n_measures = 0
    repeats = False
    staves_total = 0

    for p_idx, part in enumerate(root.iter("part")):
        divisions = num(1)
        start = num(0)
        staves_total_part = 1
        for m_idx, measure in enumerate(part.iter("measure")):
            if p_idx == 0:
                n_measures += 1
            cursor = num(0)
            reach = num(0)
            last_onset = num(0)
            for el in measure:
                tag = el.tag
                if tag == "attributes":
                    divisions = num(el.findtext("divisions") or divisions)
                    staves_total_part = max(staves_total_part, int(el.findtext("staves") or 1))
                    for t in el.iter("time"):
                        sig = "senza-misura" if t.find("senza-misura") is not None else \
                            f"{t.findtext('beats')}/{t.findtext('beat-type')}"
                        time_sigs.append((start + cursor, sig))
                    for k in el.iter("key"):
                        if k.find("fifths") is not None:
                            key_sigs.append((start + cursor, int(k.findtext("fifths")), k.findtext("mode") or ""))
                elif tag == "backup":
                    cursor -= num(el.findtext("duration") or 0) / divisions
                elif tag == "forward":
                    cursor += num(el.findtext("duration") or 0) / divisions
                    reach = max(reach, cursor)
                elif tag in ("direction", "sound"):
                    snd = el if tag == "sound" else el.find("sound")
                    if snd is not None:
                        if snd.get("tempo"):
                            try:
                                tempos.append((start + cursor, float(snd.get("tempo"))))
                            except ValueError:
                                pass
                        if snd.get("dacapo") or snd.get("dalsegno"):
                            repeats = True
                elif tag == "barline":
                    if el.find("repeat") is not None or el.find("ending") is not None:
                        repeats = True
                elif tag == "note":
                    is_chord = el.find("chord") is not None
                    is_grace = el.find("grace") is not None
                    dur = num(0) if is_grace else num(el.findtext("duration") or 0) / divisions
                    onset = last_onset if is_chord else cursor
                    pitch = el.find("pitch")
                    if pitch is not None:
                        if is_grace:
                            grace += 1
                        else:
                            midi = 12 * (int(pitch.findtext("octave")) + 1) + STEP[pitch.findtext("step")] + \
                                   float(pitch.findtext("alter") or 0)
                            ties = [t.get("type") for t in el.findall("tie")]
                            attack = "stop" not in ties
                            staff = int(el.findtext("staff") or 1)
                            voice = el.findtext("voice") or "1"
                            notes.append((p_idx, staff, voice, start + onset, dur, midi, attack))
                            measure_voices[(p_idx, staff, m_idx)].add(voice)
                            if attack and el.findtext("type"):
                                note_types.append(el.findtext("type"))
                    if not is_chord:
                        last_onset = cursor
                        cursor += dur
                    reach = max(reach, cursor)
            # parts after the first follow the longest measure lengths seen so far
            measure_len[m_idx] = max(measure_len[m_idx], reach)
            start += reach if p_idx == 0 else measure_len[m_idx]
        staves_total += staves_total_part
    return root, notes, note_types, grace, tempos, measure_len, measure_voices, time_sigs, key_sigs, n_measures, \
        repeats, staves_total


def seconds_at(tempos, total_q):
    """Written duration in seconds, integrating the stored tempo over quarter-note offsets."""
    events = sorted(tempos)
    if not events or events[0][0] > 0:
        events = [(0.0, DEFAULT_TEMPO)] + events
    secs = 0.0
    for i, (t0, qpm) in enumerate(events):
        t1 = events[i + 1][0] if i + 1 < len(events) else total_q
        if t1 > t0 and qpm > 0:
            secs += (t1 - t0) * 60.0 / qpm
    return secs


def entropy(values):
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def count_tags(root):
    c = defaultdict(int)
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue
        if tag == "dynamics":
            c["dynamics"] += 1
        elif tag == "wedge" and el.get("type") in ("crescendo", "diminuendo"):
            c["hairpins"] += 1
        elif tag == "articulations":
            c["articulations"] += len(el)
        elif tag == "ornaments":
            c["ornaments"] += sum(1 for x in el if x.tag != "accidental-mark")
        elif tag == "pedal" and el.get("type") in ("start", "sostenuto", "resume"):
            c["pedal"] += 1
        elif tag == "fingering" and (el.text or "").strip():
            c["fingering"] += 1
        elif tag == "lyric":
            c["lyrics"] += 1
        elif tag == "slur" and el.get("type") == "start":
            c["slurs"] += 1
        elif tag == "harmony":
            c["chord_symbols"] += 1
        elif tag == "metronome":
            c["metronome_marks"] += 1
        elif tag == "words":
            c["text_directions"] += 1
    return c


def analyse(file_name):
    path = os.path.join(SCORES, file_name)
    row = {"file_name": file_name}
    try:
        (root, notes, note_types, grace, tempos, measure_len, measure_voices, time_sigs, key_sigs, n_measures,
         repeats, staves_total) = parse(path)
    except Exception as e:  # noqa: BLE001 - report and keep going
        row["error"] = f"{type(e).__name__}: {e}"
        return row, {"file_name": file_name}, {"file_name": file_name}
    meta = {"file_name": file_name, **metadata(root)}

    total_q = sum(measure_len.values())
    secs = seconds_at(tempos, total_q)
    arr = np.array([(n[3], n[4], n[5], n[6]) for n in notes], dtype=float) if notes else np.zeros((0, 4))
    onset, dur, midi, attack = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3].astype(bool)
    att_on = np.unique(np.round(onset[attack], 6))

    # polyphony: notes sounding at each attack instant, over all staves
    starts, ends = np.sort(onset), np.sort(onset + dur)
    sounding = np.searchsorted(starts, att_on + 1e-9, side="right") - np.searchsorted(ends, att_on + 1e-9,
                                                                                         side="right")
    # chord span: pitch spread of notes struck together on one staff
    groups = defaultdict(list)
    for (p, s, v, on, d, m, a) in notes:
        if a:
            groups[(p, s, round(on, 6))].append(m)
    spans = np.array([max(g) - min(g) for g in groups.values() if len(g) > 1])
    sizes = np.array([len(g) for g in groups.values()])

    voices_per_staff = [len(v) for v in measure_voices.values()]
    shortest = max(note_types, key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else -1) if note_types else ""
    tempo_vals = [q for _, q in tempos]
    first_ts = min(time_sigs)[1] if time_sigs else ""
    first_key = min(key_sigs) if key_sigs else None

    row.update({
        "notes": int(len(notes)),
        "attacks": int(attack.sum()),
        "grace_notes": grace,
        "measures": n_measures,
        "parts": len(root.findall("part")),
        "staves": staves_total,
        "written_quarters": round(total_q, 3),
        "has_stored_tempo": bool(tempos),
        "first_tempo_qpm": tempos[0][1] if tempos else DEFAULT_TEMPO,
        "tempo_changes": len(set(tempo_vals)) - 1 if tempo_vals else 0,
        "seconds": round(secs, 2),
        "has_repeats": repeats,
        "onsets_per_min": round(len(att_on) / (secs / 60), 2) if secs > 0 else np.nan,
        "notes_per_sec": round(attack.sum() / secs, 3) if secs > 0 else np.nan,
        "pitch_min": float(midi.min()) if len(midi) else np.nan,
        "pitch_max": float(midi.max()) if len(midi) else np.nan,
        "pitch_range": float(midi.max() - midi.min()) if len(midi) else np.nan,
        "pitch_entropy": entropy(np.round(midi[attack])) if attack.any() else np.nan,
        "pc_entropy": entropy(np.round(midi[attack]) % 12) if attack.any() else np.nan,
        "polyphony_mean": round(float(sounding.mean()), 3) if len(sounding) else np.nan,
        "polyphony_max": int(sounding.max()) if len(sounding) else 0,
        "chord_size_max": int(sizes.max()) if len(sizes) else 0,
        "chord_share": round(float((sizes > 1).mean()), 4) if len(sizes) else np.nan,
        "chord_span_max": float(spans.max()) if len(spans) else 0.0,
        "chord_span_p95": float(np.percentile(spans, 95)) if len(spans) else 0.0,
        "chords_over_octave": round(float((spans > 12).mean()), 4) if len(spans) else 0.0,
        "voices_max": max(voices_per_staff) if voices_per_staff else 0,
        "multivoice_share": round(float(np.mean([v > 1 for v in voices_per_staff])), 4) if voices_per_staff else 0,
        "shortest_type": shortest,
        "shortest_dur_q": float(dur[attack & (dur > 0)].min()) if (attack & (dur > 0)).any() else np.nan,
        "time_sig_first": first_ts,
        "time_sigs_distinct": len({s for _, s in time_sigs}),
        "time_sig_changes": sum(a[1] != b[1] for a, b in zip(sorted(time_sigs), sorted(time_sigs)[1:])),
        "key_fifths_first": first_key[1] if first_key else np.nan,
        "key_mode_first": first_key[2] if first_key else "",
        "key_sigs_distinct": len({k for _, k, _ in key_sigs}),
    })
    row.update(count_tags(root))
    keys, counts = np.unique(np.round(midi[attack]).astype(int), return_counts=True)
    key_counts = {"file_name": file_name, **{str(k): int(n) for k, n in zip(keys, counts)}}
    return row, meta, key_counts


def main():
    files = sorted((f for f in os.listdir(SCORES) if f.endswith(".musicxml")),
                   key=lambda f: int(f.split("_")[1].split(".")[0]))
    with Pool(int(os.environ.get("WORKERS", os.cpu_count() or 4))) as pool:
        out = []
        for i, res in enumerate(pool.imap(analyse, files, chunksize=4), 1):
            out.append(res)
            if i % 100 == 0:
                print(f"{i}/{len(files)}", file=sys.stderr)
    os.makedirs(OUT, exist_ok=True)
    content = pd.DataFrame([r for r, _, _ in out]).fillna({k: 0 for k in (
        "dynamics", "hairpins", "articulations", "ornaments", "pedal", "fingering", "lyrics", "slurs",
        "chord_symbols", "metronome_marks", "text_directions")})
    content.to_csv(os.path.join(OUT, "content.csv"), index=False)
    pd.DataFrame([m for _, m, _ in out]).to_csv(os.path.join(OUT, "metadata.csv"), index=False)
    keys = pd.DataFrame([k for _, _, k in out]).set_index("file_name").fillna(0).astype(int)
    piano = [str(k) for k in range(21, 109)]
    keys = keys.reindex(columns=piano, fill_value=0).assign(
        out_of_range=keys.drop(columns=[c for c in piano if c in keys]).sum(axis=1))
    keys.to_csv(os.path.join(OUT, "key_counts.csv"))
    print(f"wrote {len(content)} rows; errors: {content.get('error', pd.Series(dtype=str)).notna().sum()}")


if __name__ == "__main__":
    main()
