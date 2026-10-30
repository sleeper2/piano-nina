"""Hand-checked values for two small test scores (FEATURES_SPEC.md section 11). Run: python nina_stats/tests/test_features.py

two_hands.musicxml, one 4/4 measure at 60 qpm. RH: C5, D#5, C5+G5, C6 (quarters). LH: C3, G2 (halves).
triplets_vs_16ths.musicxml, two 4/4 measures. m1: RH six triplet eighths + half, LH four 16ths + dotted half.
m2: both hands four quarters.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import coordination  # noqa: E402
import hand_features  # noqa: E402

hand_features.SCORES = coordination.SCORES = HERE


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-9), (a, b)


def test_two_hands():
    r, ms = hand_features.features("two_hands.musicxml")
    close(r["seconds"], 4.0)
    assert r["hand_confidence"] == "staff"
    close(r["simultaneous_pitch_span_mean"], (24 + 0 + 36 + 0) / 4)  # onsets 0, 1, 2, 3
    close(r["simultaneous_pitch_span_ge24"], 0.5)
    close(r["hand_pitch_span_max"], 7)                                # C5+G5
    close(r["black_key_note_ratio"], 1 / 7)                           # D#5 of 7 notes
    close(r["black_key_attack_ratio_rh"], 1 / 4)
    close(r["black_key_notes_per_second"], 1 / 4)
    close(r["rh_displacement_mean"], (3 + 0.5 + 8.5) / 3)             # medians 72, 75, 75.5, 84
    close(r["lh_displacement_mean"], 5)                               # 48 -> 43
    close(r["hand_displacement_semitones_gt7"], 1 / 4)
    close(r["hand_displacement_per_quarter"], 17 / 4)
    assert r["hand_crossing_count"] == 0
    close(ms[0]["displacement_sum"], 17)
    c, _ = coordination.features("two_hands.musicxml")
    close(c["cross_hand_simultaneous_onset_ratio"], 2 / 4)
    close(c["cross_hand_simultaneous_onset_ratio_lh"], 1)
    close(c["cross_hand_simultaneous_onset_ratio_rh"], 2 / 4)
    close(c["cross_hand_onset_alignment_tol0"], 4 / 6)
    close(c["onset_density_diff"], (4 - 2) / 6)


def test_triplets_vs_16ths():
    r, ms = coordination.features("triplets_vs_16ths.musicxml")
    close(r["cross_hand_simultaneous_onset_ratio"], 6 / 14)
    assert r["cross_hand_alternations"] == 4                         # 1/4 L, 1/3 R, 1/2 L, 2/3 R, 3/4 L
    close(r["cross_hand_alternation_switch_share"], 4 / 13)
    assert r["cross_hand_alternation_longest_run"] == 5
    close(r["cross_hand_onset_alignment_tol010"], (8 + 8) / 20)
    close(r["cross_hand_onset_density_ratio"], 11 / 9)
    close(r["cross_hand_simultaneous_activity_ratio"], 1)
    close(r["cross_hand_subdivision_mismatch_cross_rhythm_share"], 0.5)
    m1, m2 = ms
    assert (m1["rh_grid"], m1["lh_grid"], m1["cross_rhythm"]) == (3, 4, True)
    assert (m2["rh_grid"], m2["lh_grid"], m2["grids_differ"]) == (1, 1, False)
    close(m1["similarity"], 2 / math.sqrt(7 * 5))
    close(m1["one_hand_share"], 8 / 10)
    close(m1["subdivision_gap"], abs(math.log2(3 / 4)))


if __name__ == "__main__":
    test_two_hands()
    test_triplets_vs_16ths()
    print("all feature checks passed")
