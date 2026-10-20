# PP1 Synthetic Motor Dataset V3 — preparation only

**Synthetic within-generator benchmark. No clinical labels. No training run.**

Start with the [engine/V2 audit](docs/ENGINE_AND_V2_AUDIT.md),
[simulation assumptions](docs/SIMULATION_AND_LIMITATIONS.md),
[frozen evaluation design](configs/evaluation_protocol_v3.json), and
[Colab notebook](notebooks/Generate_and_Audit_V3.ipynb).

The [generator](pp1_v3.py) preserves the [original engine](../reference_engine/pp1_motor_engine.py)
by pinning its SHA256. Full generation requires both Colab CPU and explicit approval.
No existing datasets are written. Source/reference numbers are not silently reused.

## Colab workflow

1. Copy this research folder and the existing V2 folder to a private Google Drive
   folder, preserving `motor_scoring_research/reference_engine`,
   `motor_scoring_research/v3`, and sibling `PADS + MJFF` paths. Copy, do not move originals.
2. Open the notebook in VS Code and select the Google Colab extension's CPU runtime.
   The notebook runs in Colab; it does not install dependencies on Windows.
3. Run setup/mount cells and edit ROOT to that Drive folder. Check the pinned engine.
4. Only after approval, enable the V2 verification switch, run it, and inspect its
   report. Mismatches block the next step and need investigation, not relabeling.
5. Only after approval, enable full generation. A unique /content output is created.
   The generator scores each session and audits serialized recordings automatically.
6. Inspect integrity/class-coverage reports. No guarantee of all five classes is
   made. Do not regenerate until attractive class counts or accuracy appear.
7. Back up the versioned output to a fresh Drive directory. No model training cells
   are present. Optional stress generation has its own disabled approval switch.

CLI equivalent (Colab only; after explicit user approval):

```bash
python pp1_v3.py generate --output /content/PP1_Synthetic_Motor_Dataset_V3_RUN_ID --approve-generation
```

The notebook constructs an absolute command with the correct engine/config paths.
For independent integrity rechecking use `audit --output DATASET_DIRECTORY`.
For V2 reproduction use [verify_v2.py](verify_v2.py); its output report path must be new.

## Generated artifacts

- `cases/<session_id>/`: four CSVs; signal values written at float64 round-trip precision.
- `splits.csv`: fixed session, seed, profile and family assignments.
- `sessions.jsonl`: per-session parameters, raw hashes, scores, MPS, code/config/engine hashes.
- `features.csv`: session join key and the nine allowlisted features only.
- `targets.csv`: four scores and demo MPS only, plus join key.
- `source_manifest.json`, `feature_schema.json`, `config.json`, `evaluation_protocol.json`.
- `dataset_manifest.json`: seed, versions, hashes, scientific claim limits.
- `code/`: exact generator and unchanged engine snapshots.
- `preparation/`: complete frozen preparation package and engine, preserving paths
  for reproducible reruns in a fresh approved Colab output directory.
- `integrity_report.json`: completeness, class counts by split, missing/rare classes,
  raw validation, duplicate checks, serialized-feature/score reproduction.
- `COMPLETE.json`: present only after the integrity audit passes.

The checked-in [session plan](manifests/session_plan_v3.csv) is metadata only, not
generated signal data. A separate [stress plan](manifests/stress_plan_v3.csv) is also
predeclared. The generator verifies these plans for the default configuration.

## Lightweight local tests

From the workspace, using already installed NumPy/SciPy:

```powershell
python -B -m unittest discover -s motor_scoring_research/v3/tests -v
```

No dependency installation is needed locally. These tests do not call the full
session synthesizer, extract V2 features, or fit/evaluate models. Passing them is
not an end-to-end generation result. See [local check results](docs/LOCAL_CHECKS.md).
