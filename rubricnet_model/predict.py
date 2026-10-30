"""Run the pretrained RubricNet difficulty model (Ramoneda et al., ISMIR 2024) on every score.

Feature extraction (music21, the slow and fragile part) runs per score in a child process
(`predict.py --one <xml> <out.json>`), so a crash or hang on one score does not take the whole run
down. The parent loads the 5 camera-ready fold models once and predicts from the 12 extracted
descriptors. Results are appended to results/predictions.jsonl; re-running skips scores that
already succeeded. With TEMPO=explicit only scores with an explicit tempo are processed (see tempo.py);
otherwise every score is (TEMPO=marked filters in evaluate.py, after extraction shows which scores have a tempo).

Per score:
  features        - the 12 RubricNet descriptors (upstream `difficulty_descriptors`)
  folds           - per fold model: ordinal class 0-8 and the rubric score scaled to 0-12 (as upstream)
  rubricnet       - mean of the 5 fold classes (evaluate.py rounds it half-up to a class)
"""
import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tempo

SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/data/score_musicxml"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
RUBRICNET_DIR = Path(os.environ.get("RUBRICNET_DIR", "/opt/rubricnet"))
CHECKPOINTS = RUBRICNET_DIR / "checkpoints" / "rubricnet_cameraready"
PREDICTIONS = RESULTS_DIR / "predictions.jsonl"
WORKERS = int(os.environ.get("WORKERS", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
TIMEOUT = int(os.environ.get("SCORE_TIMEOUT", "1800"))
# Tempo assumed when a score has no metronome mark (upstream predict.py and the CIPI feature run use 100).
DEFAULT_BPM = 100
# With more than two staves, staves holding less than this share of the notes are skipped (see README).
MIN_STAFF_SHARE = 0.1


def load_upstream():
    """Upstream predict.py, imported under another name (this file is also called predict.py)."""
    spec = importlib.util.spec_from_file_location("rubricnet_upstream", RUBRICNET_DIR / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_one(xml_path, out_path):
    """Same as upstream `extract_features` + `difficulty_descriptors`, except that exceptions are
    raised instead of swallowed and the RH/LH staves are chosen by `StaffSelectingExtractor`."""
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from music21 import stream
    from music21.tempo import MetronomeMark
    from extractor.DifficultyFeatures.code.raw_data_extractors import CIPIFeatureExtractor, convert_offsets

    upstream = load_upstream()
    chosen = {}

    class StaffSelectingExtractor(CIPIFeatureExtractor):
        """Upstream takes staff 0 as RH and staff 1 as LH. With more than two staves (an ossia part,
        a 3-4 staff layout), near-empty staves are skipped first. Two-staff scores are unchanged."""

        def extract_rh_lh(self, m21_stream):
            staves = list(m21_stream.getElementsByClass(stream.Part))
            counts = [len(s.recurse().notes) for s in staves]
            keep = list(range(len(staves)))
            if len(staves) > 2:
                big = [i for i, n in enumerate(counts) if n >= MIN_STAFF_SHARE * sum(counts)]
                if len(big) >= 2:
                    keep = big
            chosen.update(n_staves=len(staves), notes_per_staff=counts, staves_used=keep[:2])
            return staves[keep[0]], staves[keep[1]]

    warnings_set = set()
    raw = StaffSelectingExtractor().extract_raw_data_map(
        [xml_path], warnings_set, metronome_mark=MetronomeMark(DEFAULT_BPM), ignore_warnings=True)
    convert_offsets(raw)
    features = upstream.difficulty_descriptors(raw)
    features = {k: float(v) for k, v in features.items()}
    bad = [k for k, v in features.items() if not np.isfinite(v)]
    if len(features) != 12 or bad:
        raise RuntimeError(f"incomplete descriptors: got {sorted(features)}, non-finite {bad}")
    result = {
        "features": features,
        "score_has_tempo": bool(raw["original_bpm"]),
        "bpm": float(raw["bpm"]),
        "cant_expand_repeats": bool(raw["cant_expand"]),
        "extractor_warnings": sorted(warnings_set),
        **chosen,
    }
    Path(out_path).write_text(json.dumps(result))


class Models:
    """The 5 camera-ready fold models with their feature scalers, loaded as upstream
    `load_model_and_predict` does (minus its hard-coded checkpoint path)."""

    def __init__(self):
        import warnings
        warnings.filterwarnings("ignore")
        import pandas as pd
        from rubricnet.rubricnet import RubricnetSklearn

        self.upstream = load_upstream()
        self.folds = []
        for split in range(5):
            scaler = pd.read_pickle(CHECKPOINTS / f"scaler_{split}.pkl")
            clf = RubricnetSklearn(input_dim=len(scaler.feature_names_in_), num_classes=9,
                                   args=self.upstream.FakeArgs(), logging=False)
            clf.load_model(str(CHECKPOINTS / f"split_{split}.ckpt"))
            self.folds.append((clf, scaler))

    def predict(self, features):
        import pandas as pd
        folds = []
        for clf, scaler in self.folds:
            x = scaler.transform(pd.DataFrame([features])[list(scaler.feature_names_in_)])
            folds.append({
                "class": int(clf.predict(x)[0].item()),
                "score": float(self.upstream.scale_to_0_12(clf.predict_regression(x).item())),
            })
        return {"folds": folds, "rubricnet": sum(f["class"] for f in folds) / len(folds)}


def prepare_score(src, dst):
    """Copy the score, fixing inputs music21 / the extractor cannot handle (scores needing no fix are
    copied unchanged). Returns the list of fixes applied.

    - <harmony> (chord symbols): music21 fails on some ("degree-value missing"); they carry no
      playable notes.
    - parts without pitched notes (an empty duplicate piano part): music21 fails to convert the score.
    - no time signature anywhere: the extractor raises "No time signature", so an invisible 4/4
      (MusicXML default) is added.
    - empty <fingering/>: music21 reads it as finger None, and sorting the finger set next to
      numbered fingers fails.
    - a rest display position that is not a pitch (<display-step>?): music21 fails to parse it.
      It only sets where the rest is drawn.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    fixes = []

    harmonies = [(parent, h) for parent in root.iter() for h in parent.findall("harmony")]
    for parent, h in harmonies:
        parent.remove(h)
    if harmonies:
        fixes.append(f"removed {len(harmonies)} chord symbols")

    parts = root.findall("part")
    empty = [p for p in parts if all(n.find("rest") is not None for n in p.iter("note"))]
    if empty and len(empty) < len(parts):
        part_list = root.find("part-list")
        kept_measures = next(p for p in parts if p not in empty).findall("measure")
        moved = 0
        for p in empty:
            # Tempo marks at the start of a measure move to the same measure of the first kept part
            # (score_40 has its only metronome mark in the empty part).
            for measure, target in zip(p.findall("measure"), kept_measures):
                for e in list(measure):
                    if e.tag in ("note", "backup", "forward"):
                        break
                    if e.tag == "direction" and (e.find(".//metronome") is not None
                                                 or e.find("sound[@tempo]") is not None):
                        measure.remove(e)
                        target.insert(0, e)
                        moved += 1
        for p in empty:
            root.remove(p)
            for sp in part_list.findall("score-part"):
                if sp.get("id") == p.get("id"):
                    part_list.remove(sp)
        fixes.append(f"removed empty parts {[p.get('id') for p in empty]}"
                     + (f", moved {moved} tempo marks from them" if moved else ""))

    if root.find(".//time") is None:
        time_xml = ET.fromstring('<time print-object="no"><beats>4</beats><beat-type>4</beat-type></time>')
        for p in root.findall("part"):
            measure = p.find("measure")
            attributes = measure.find("attributes")
            if attributes is None:
                attributes = ET.Element("attributes")
                measure.insert(0, attributes)
            # MusicXML order: divisions, key, time, ...
            preceding = [i for i, e in enumerate(attributes) if e.tag in ("divisions", "key")]
            attributes.insert(preceding[-1] + 1 if preceding else 0, copy.deepcopy(time_xml))
        fixes.append("added 4/4 time signature (score has none)")

    fingerings = [(parent, f) for parent in root.iter() for f in parent.findall("fingering")
                  if not (f.text or "").strip()]
    for parent, f in fingerings:
        parent.remove(f)
    if fingerings:
        fixes.append(f"removed {len(fingerings)} empty fingerings")

    bad_rests = [rest for rest in root.iter("rest")
                 if rest.find("display-step") is not None
                 and (rest.findtext("display-step") or "").strip() not in tuple("ABCDEFG")]
    for rest in bad_rests:
        for tag in ("display-step", "display-octave"):
            for e in rest.findall(tag):
                rest.remove(e)
    if bad_rests:
        fixes.append(f"removed {len(bad_rests)} invalid rest display positions")

    if fixes:
        tree.write(dst, encoding="UTF-8", xml_declaration=True)
    else:
        shutil.copy(src, dst)
    return fixes


def run_child(xml_file):
    started = time.time()
    record = {"file_name": xml_file.name}
    with tempfile.TemporaryDirectory() as tmp:
        local_xml = Path(tmp) / xml_file.name
        fixes = prepare_score(xml_file, local_xml)
        if fixes:
            record["input_fixes"] = fixes
        out_json = Path(tmp) / "out.json"
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--one", str(local_xml), str(out_json)],
                cwd=tmp, capture_output=True, text=True, timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            record.update(status="error", error=f"timeout after {TIMEOUT}s")
        else:
            if proc.returncode == 0 and out_json.exists():
                record.update(status="ok", **json.loads(out_json.read_text()))
            else:
                lines = (proc.stderr + proc.stdout).strip().splitlines()
                # Upstream prints swallowed exceptions (e.g. a failed music21 parse) before ours.
                exceptions = [ln for ln in lines if re.match(r"[\w.]+(Error|Exception)\b", ln)]
                tail = list(dict.fromkeys(exceptions[-3:] + lines[-2:]))
                record.update(status="error", error=f"exit {proc.returncode}: " + " | ".join(tail))
    record["seconds"] = round(time.time() - started, 1)
    return record


def load_done():
    done = {}
    if PREDICTIONS.exists():
        for line in PREDICTIONS.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["file_name"]] = rec
    return {name for name, rec in done.items() if rec["status"] == "ok"}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_files = sorted(SCORES_DIR.glob("*.musicxml"), key=lambda p: int(p.stem.split("_")[-1]))
    files = tempo.selected(all_files)
    if LIMIT:
        files = files[:LIMIT]
    done = load_done()
    todo = [f for f in files if f.name not in done]
    # Longest scores first so they do not end up as stragglers at the end of the run.
    todo.sort(key=lambda p: p.stat().st_size, reverse=True)
    print(f"TEMPO={tempo.TEMPO}: {len(files)} of {len(all_files)} scores, {len(done)} already done, "
          f"{len(todo)} to process, {WORKERS} workers")
    if not todo:
        return
    models = Models()

    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, PREDICTIONS.open("a") as out:
        futures = [pool.submit(run_child, f) for f in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec["status"] == "ok":
                rec.update(models.predict(rec["features"]))
            out.write(json.dumps(rec) + "\n")
            out.flush()
            summary = (f"rubricnet={rec['rubricnet']:.1f}" if rec["status"] == "ok" else rec["error"][:200])
            print(f"[{i}/{len(todo)}] {rec['file_name']} {rec['status']} {rec['seconds']}s "
                  f"{summary} (elapsed {time.time() - started:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        extract_one(sys.argv[2], sys.argv[3])
    else:
        main()
