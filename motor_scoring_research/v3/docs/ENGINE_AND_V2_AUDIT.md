# Engine and V2 source audit — 2026-10-21

Inputs: [original engine](../../reference_engine/pp1_motor_engine.py),
[V2 generator](../../../PADS%20+%20MJFF/regenerate_dataset.py),
[V2 features](../../../PADS%20+%20MJFF/features_extracted_by_existing_PP1_engine.csv).
Neither input was edited or executed as a generator.

Engine SHA256: `c39620fe8738ca9217b1b7f9d75141e0f2ab9a7b49d22ebd63b6234202c437e4`.
V2 generator SHA256: `adef793184258e971659c001e389556786d79ff74955dd4f6cf273a754184d69`.

## Exact extraction and thresholds

Inputs must have finite values and shapes TS=(1000,3), RS=(1000,3), GI=(18000,3) at 100 Hz.
BS is a finite 1D strictly increasing event list in [0,10). The engine receives arrays;
it does not validate CSV timestamps. The V3 adapter adds that validation.

| Domain | Extraction | Score 0 / 1 / 2 / 3 / 4 |
|---|---|---|
| TS | sqrt(mean((norm(acceleration)-9.81)^2)); no bandpass, detrending or gravity-vector estimation | <.12 / [.12,.24) / [.24,.40) / [.40,.70) / >=.70 m/s² |
| BS | count*6 taps/min; population std of adjacent tap intervals, null if <2 events | >=210 / [165,210) / [120,165) / [75,120) / <75 taps/min |
| RS | norm(gyro); scipy find_peaks distance=45 samples, prominence=.2; mean detected peak magnitude; null if no peaks | >=2.4 / [1.8,2.4) / [1.2,1.8) / [.6,1.2) / <.6 rad/s; null remains null |
| GI | norm(acceleration); third-order Butterworth .6–3 Hz bandpass, sosfiltfilt; peaks distance=40 samples, prominence=.3; count/3 steps/min; population std of peak intervals | >=110 / [95,110) / [80,95) / [65,80) / <65 candidates/min |

BS interval irregularity does not affect its current score. Neither do RS peak count
or GI interval irregularity. TS is affected by bias and non-tremor acceleration.
GI detects wrist candidate peaks, not verified footfalls. Zero detected steps receive
GI=4 in this unchanged engine; this is not proof of gait impairment. No RS peaks
yield RS=null and MPS=null. Real mode always returns features with scores/MPS=null.

## Why V2 TS=4 is absent

The inspected stored TS feature range is 0.0215664–0.64770839 m/s²; none reaches .70.
V2's strongest amplitude parameter is 1.32*[.88,1.18], or 1.1616–1.5576 m/s².
Its gravity-axis tremor component is only .68 times that amplitude, multiplied by
an envelope `.85 + .16*sin(...)`. To first order, magnitude-minus-gravity follows
the gravity-axis component. Its expected RMS is approximately
`A*.68*sqrt((.85²+.16²/2)/2)`, about .643 at the maximum A. Transverse components,
noise and finite-window phase affect this approximation; it is not an exact bound.
The measured stored maximum explains absence, without changing the thresholds.

A pure gravity-axis sinusoid with peak acceleration 1.2 m/s² at 4 Hz has RMS
.8485 m/s² and receives TS=4. Its equivalent translational displacement amplitude
is A/(2*pi*f)² = 1.90 mm. V3's maximum 2.6 m/s² at 4 Hz corresponds to 4.12 mm.
These are mechanically feasible illustrative motions, not clinical severity norms
or empirically validated tremor amplitudes. Orientation relative to gravity matters;
rotating gravity and motion together leaves the engine's magnitude unchanged.
The small fixture tests the formula, not a full assessment or clinical validation.

## Reproduction status

The engine's feature names and call signature match V2's external-engine interface.
V2 writes features/scores before rounding raw samples to six decimals. Therefore
raw CSV recomputation may differ slightly, or occasionally change peak detections.
Full reproduction has NOT run locally. [verify_v2.py](../verify_v2.py) performs
all 200 sessions only in Colab, compares integer counts and labels exactly, reports
per-feature maximum errors, and uses 2e-6 absolute tolerance for continuous features.
Mismatches stop the notebook for investigation; they are never silently relabeled.
This gate must pass before claiming the supplied engine reproduces V2.

## V2 sources and leakage

V2 uses 51 reference records according to its manifest (23 control-labeled, 28
PD-labeled). The original table is absent. The code samples individual frequency
and RMS arrays independently with replacement, then mixes gait frequency with a
hand-selected severity target (.72 simulated target + .28 reference frequency).
Thus the documentation's “aggregate” description omits sampling from individual
numeric values; a frequency/RMS pair need not belong to one person.
The amplitude multiplier, clipping, target frequencies, phase/noise equations,
TS amplitudes/frequencies, tap settings and RS settings are simulation assumptions.

Existing session split is 140/30/30, with 70/15/15 from each synthetic profile.
No session-ID overlap was found. Train/validation share 22 gait frequency values
and 24 RMS values; train/test share 20 and 23 respectively. There are no source IDs
or per-session seed records. All cases use the same generator and global random
stream. The 128 distinct four-domain level vectors include 8 shared between train
and validation and 11 between train and test. This is a within-generator split,
not proof of source or template independence; raw duplicates remain a Colab check.

V2 total class counts [0,1,2,3,4]: TS=[112,43,25,20,0]; BS=[73,38,45,28,16];
RS=[49,38,60,30,23]; GI=[65,60,42,23,10]. All four tests are simulated together per
session; none of the 200 case IDs denotes a real patient.
