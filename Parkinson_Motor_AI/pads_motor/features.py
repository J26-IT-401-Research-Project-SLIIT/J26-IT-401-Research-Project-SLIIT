"""Per-record features, independent of diagnosis label.

Raw PADS input channels order: AX AY AZ [g] GX GY GZ [rad/s].
These are engineered MOVEMENT features, not validated TS/RS/BS/GI scores.
"""
import numpy as np
import pandas as pd


def features_from_record(x: np.ndarray, sampling_hz=100.0):
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("Expected [n_samples,6] accelerometer/gyroscope")
    a = x[:, :3].astype(np.float64)
    w = x[:, 3:].astype(np.float64)
    # Center each acceleration axis by median to attenuate static orientation/gravity.
    a_dynamic = a - np.median(a, axis=0, keepdims=True)
    amag = np.linalg.norm(a_dynamic, axis=1)
    gyro_mag = np.linalg.norm(w, axis=1)
    jerk = np.diff(a_dynamic, axis=0) * sampling_hz
    jmag = np.linalg.norm(jerk, axis=1)
    feats = {
        "acc_dynamic_rms_g": float(np.sqrt(np.mean(amag ** 2))),
        "acc_dynamic_mean_g": float(amag.mean()),
        "acc_dynamic_std_g": float(amag.std()),
        "acc_dynamic_p95_g": float(np.percentile(amag, 95)),
        "acc_dynamic_peak_g": float(amag.max()),
        "jerk_rms_g_per_s": float(np.sqrt(np.mean(jmag ** 2))),
        "gyro_mean_rad_s": float(gyro_mag.mean()),
        "gyro_rms_rad_s": float(np.sqrt(np.mean(gyro_mag ** 2))),
        "gyro_std_rad_s": float(gyro_mag.std()),
        "gyro_p95_rad_s": float(np.percentile(gyro_mag, 95)),
        "gyro_peak_rad_s": float(gyro_mag.max()),
        "gyro_peak_to_peak_rad_s": float(gyro_mag.max() - gyro_mag.min()),
    }
    for i, axis in enumerate(("x", "y", "z")):
        feats[f"acc_{axis}_std_g"] = float(a_dynamic[:, i].std())
        feats[f"gyro_{axis}_std_rad_s"] = float(w[:, i].std())
    return feats


def build_features(signals, meta):
    vals = [features_from_record(s) for s in signals]
    df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(vals)], axis=1)
    return df
