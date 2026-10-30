"""Flag the scores that have one explicit, consistent tempo.

A score qualifies when, in its MusicXML:
  1. a printed metronome mark with a number (e.g. quarter = 120) is in the first measure;
  2. every metronome mark in the score has that same value (no other tempos later, no
     metric-relation marks like quarter = dotted quarter);
  3. every playback tempo (<sound tempo>) equals the printed mark converted to quarter notes per
     minute (no contradiction between printed and playback tempo).
Beat unit (half, dotted quarter, ...) and parentheses do not matter. Tempo words without a number
("rit.", "a tempo") are not tempo values and are ignored.

Output (in results/): explicit_tempo_scores.csv
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from evaluate import DATASET, RESULTS_DIR

SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/data/score_musicxml"))
UNIT_IN_QUARTERS = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "16th": 0.25, "32nd": 0.125}


def metronome_value(metronome):
    """Printed mark as (unit, dots, per-minute, quarter bpm), or None if it has no number."""
    units = metronome.findall("beat-unit")
    per_minute = metronome.findtext("per-minute")
    if len(units) != 1 or per_minute is None:
        return None
    try:
        bpm = float(per_minute.strip())
    except ValueError:
        return None
    unit = units[0].text.strip()
    dots = len(metronome.findall("beat-unit-dot"))
    factor = UNIT_IN_QUARTERS.get(unit)
    if factor is None:
        return None
    factor *= 2 - 0.5 ** dots  # 1 dot = x1.5, 2 dots = x1.75
    return unit, dots, bpm, bpm * factor


def check_tempo(path):
    root = ET.parse(path).getroot()
    marks, sounds = [], []  # (measure index, value)
    for part in root.findall("part"):
        for i, measure in enumerate(part.findall("measure")):
            for metronome in measure.iter("metronome"):
                marks.append((i, metronome_value(metronome)))
            for sound in measure.iter("sound"):
                if sound.get("tempo") is not None:
                    sounds.append((i, float(sound.get("tempo"))))

    first = [v for i, v in marks if i == 0 and v is not None]
    if not first:
        return False, "no printed numeric tempo in the first measure", None
    unit, dots, bpm, quarter_bpm = first[0]
    if any(v is None or v[:3] != (unit, dots, bpm) for _, v in marks):
        return False, "other tempo marks follow", quarter_bpm
    if any(abs(t - quarter_bpm) > 1 for _, t in sounds):
        return False, "playback tempo differs from printed tempo", quarter_bpm
    return True, "", quarter_bpm


def main():
    labels = pd.read_excel(DATASET, sheet_name="dataset")
    labels = labels.rename(columns={labels.columns[1]: "label"})
    labels["number"] = labels["file_name"].str.extract(r"(\d+)")[0].astype(int)
    labels = labels.sort_values("number")

    checks = [check_tempo(SCORES_DIR / f) for f in labels["file_name"]]
    labels["explicit_tempo"] = [c[0] for c in checks]
    labels["tempo_reason"] = [c[1] for c in checks]
    labels["quarter_bpm"] = [c[2] for c in checks]
    labels[["file_name", "label", "bpm", "fact", "explicit_tempo", "tempo_reason", "quarter_bpm"]] \
        .to_csv(RESULTS_DIR / "explicit_tempo_scores.csv", index=False)
    print(f"{int(labels['explicit_tempo'].sum())} of {len(labels)} scores have one explicit tempo")


if __name__ == "__main__":
    main()
