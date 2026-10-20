# Preparation checks — 2026-10-21

Executed locally using existing NumPy/SciPy; no dependencies installed.

`python -B -m unittest discover -s motor_scoring_research/v3/tests -v`

**16 tests passed.** Fixtures include 20 samples per waveform/domain/family,
a 25-sample TS mathematical illustration, and at most one second of tap events.
No 10-second/180-second assessment session was synthesized. No existing raw
recordings were processed, and no models were fitted or evaluated.

Covered: unchanged engine hash and rejection of a changed engine; exact threshold
boundary semantics for all domains; TS=4 formula feasibility; orientation magnitude
invariance; deterministic 240-session metadata plan with 168/36/36 assignments;
40-session stress metadata separation; parameter reproducibility; valid tiny tap
events; finite/deterministic tiny waveforms for all four generator families;
features-only real mode using a stub; missing RS keeps MPS null; exclusive writes;
approval and runtime gates; feature allowlist.

Python sources and notebook Python cells were statically parsed (the Colab `%pip`
line is excluded from Python AST parsing). Notebook metadata, empty outputs,
disabled processing switches and frozen input hashes were checked.

Not yet verified: end-to-end Colab execution, V2 raw-to-feature reproduction,
full V3 raw integrity, score distributions, coverage of all five classes, actual
runtime/resource usage, or any ML performance. These require explicit approval
and Colab. Unit tests are not evidence of clinical realism or model accuracy.
