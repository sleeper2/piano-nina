"""Run the pretrained CIPI difficulty models (Ramoneda et al., ESWA 2023) on every score.

Each score is processed in its own child process (`predict.py --one <xml> <out.json>`) so that
a crash, OOM kill or hang on one score does not take the whole run down. Results are appended
to results/predictions.jsonl; re-running skips scores that already succeeded (a score where
only VirtuosoNet failed is retried).

Models (each is 5 cross-validation folds, fold predictions are averaged as upstream does):
  p           - piano-roll pitch representation
  argnn       - ArGNN piano-fingering embeddings
  virtuoso    - VirtuosoNet expressive-performance embeddings
  ensemble    - mean of the three (upstream `compute_difficulty`)
  argnn_fixed - argnn with correct right-hand sequence length (see README, not in ensemble)
"""
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/data/score_musicxml"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PREDICTIONS = RESULTS_DIR / "predictions.jsonl"
WORKERS = int(os.environ.get("WORKERS", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
TIMEOUT = int(os.environ.get("SCORE_TIMEOUT", "3600"))


def run_virtuoso(xml_path, device):
    """Same as upstream `compute_difficulty_virtuoso`, but via subprocess.run instead of os.system,
    so a failure of the VirtuosoNet script reports its stderr instead of a missing-file error."""
    import torch
    from eswa_difficulty import compute_difficulty as cd

    embedding_path = xml_path + ".pt"
    proc = subprocess.run(
        [sys.executable, cd.get_package_resource("virtuosoNet/virtuoso"), "--session_mode=inference",
         f"--checkpoint={cd.get_package_resource('virtuosoNet/checkpoint_last.pt')}",
         f"--xml_path={xml_path}", f"--save_embedding={embedding_path}"],
        capture_output=True, text=True,
    )
    if not os.path.exists(embedding_path):
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"VirtuosoNet exit {proc.returncode}: " + " | ".join(tail))
    emb = torch.load(embedding_path)["total_note_cat"].transpose(0, 1).squeeze()
    os.remove(embedding_path)
    with torch.no_grad():
        return cd.get_pred(emb, device, cd.get_models(device, "virtuoso"), "virtuoso")


def predict_one(xml_path, out_path):
    import torch
    from eswa_difficulty import compute_difficulty as cd
    from eswa_difficulty.piano_fingering.compute_embeddings import compute_embedding_score

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    device = "cpu"
    result = {}

    # One fingering embedding pass shared by the `p` and `argnn` models
    # (upstream computes it twice with identical output).
    feats = compute_embedding_score("ArGNNThumb-s", xml_path)
    if not feats:
        raise RuntimeError("fingering embedding failed (score parsing error)")

    pitches, _ = cd.get_pitches(feats)
    result["p_folds"] = cd.get_pred(pitches, device, cd.get_models(device, "p"), "p")

    argnn_models = cd.get_models(device, "argnn")
    result["argnn_folds"] = cd.get_pred(cd.get_argnn(feats), device, argnn_models, "argnn")

    argnn = cd.get_argnn(feats)
    length = cd.torch_tuple(torch.Tensor([argnn.rh.shape[0]]), torch.Tensor([argnn.lh.shape[0]]))
    argnn.unsqueeze(0)
    with torch.no_grad():
        result["argnn_fixed_folds"] = [
            cd.prediction2label(m(argnn, length)).tolist()[0] for m in argnn_models
        ]

    # The VirtuosoNet script writes `<xml>.pt` next to the score, which is why the score is
    # copied to a temp dir first.
    try:
        result["virtuoso_folds"] = run_virtuoso(xml_path, device)
    except Exception as e:  # keep p/argnn results even if VirtuosoNet fails on this score
        result["virtuoso_folds"] = []
        result["virtuoso_error"] = f"{type(e).__name__}: {e}"

    for name in ("p", "argnn", "argnn_fixed", "virtuoso"):
        result[name] = cd.average_difficulty(result[f"{name}_folds"])
    result["ensemble"] = cd.average_difficulty([result["p"], result["argnn"], result["virtuoso"]])
    result["n_notes_rh"] = int(len(feats["rh"]["pitches"]))
    result["n_notes_lh"] = int(len(feats["lh"]["pitches"]))

    Path(out_path).write_text(json.dumps(result))


def prepare_score(src, dst):
    """Copy the score, fixing inputs the upstream parser cannot handle (scores needing no fix are
    copied unchanged). Returns the list of fixes applied.

    - <harmony> (chord symbols): ChordSymbolParseException on some; they carry no playable notes.
    - parts without pitched notes (e.g. an empty duplicate piano part): the parser adds 10 to the
      voice numbers of every later part, so all notes land in the left hand (voice >= 5) and the
      right hand is empty.
    - no time signature anywhere: the parser needs one, so an invisible 4/4 (MusicXML default) is added.
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
        for p in empty:
            root.remove(p)
            for sp in part_list.findall("score-part"):
                if sp.get("id") == p.get("id"):
                    part_list.remove(sp)
        fixes.append(f"removed empty parts {[p.get('id') for p in empty]}")

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
                lines = (proc.stderr or proc.stdout).strip().splitlines()
                # Upstream prints swallowed exceptions before ours, so keep every exception line.
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
    return {name for name, rec in done.items() if rec["status"] == "ok" and "virtuoso_error" not in rec}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SCORES_DIR.glob("*.musicxml"), key=lambda p: int(p.stem.split("_")[-1]))
    if LIMIT:
        files = files[:LIMIT]
    done = load_done()
    todo = [f for f in files if f.name not in done]
    # Longest scores first so they do not end up as stragglers at the end of the run.
    todo.sort(key=lambda p: p.stat().st_size, reverse=True)
    print(f"{len(files)} scores, {len(done)} already done, {len(todo)} to process, {WORKERS} workers")

    lock = threading.Lock()
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, PREDICTIONS.open("a") as out:
        futures = [pool.submit(run_child, f) for f in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            summary = (f"ens={rec['ensemble']:.2f}" if rec["status"] == "ok" else rec["error"][:200])
            print(f"[{i}/{len(todo)}] {rec['file_name']} {rec['status']} {rec['seconds']}s "
                  f"{summary} (elapsed {time.time() - started:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        predict_one(sys.argv[2], sys.argv[3])
    else:
        main()
