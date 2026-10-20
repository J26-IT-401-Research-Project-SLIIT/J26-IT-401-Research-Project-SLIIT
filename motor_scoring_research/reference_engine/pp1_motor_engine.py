import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

# DEMO ONLY: arbitrary software-testing thresholds.
# NOT calibrated against real devices or clinical ratings.
DEMO_THRESHOLDS = {
    "TS": [0.12, 0.24, 0.40, 0.70],
    "BS": [210, 165, 120, 75],
    "RS": [2.4, 1.8, 1.2, 0.6],
    "GI": [110, 95, 80, 65]
}

DURATIONS = {"TS": 10, "BS": 10, "RS": 10, "GI": 180}


def check_sensor(data, seconds, fs, name):
    data = np.asarray(data, dtype=float)
    expected = int(round(seconds * fs))

    if data.shape != (expected, 3):
        raise ValueError(
            f"{name}: expected ({expected}, 3) samples"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{name}: invalid sensor values")

    return data


def extract_features(ts_acc, bs_taps, rs_gyro, gi_acc, fs=100):

    if not np.isfinite(fs) or fs <= 6:
        raise ValueError("Invalid sampling rate")

    ts = check_sensor(ts_acc, 10, fs, "TS")
    rs = check_sensor(rs_gyro, 10, fs, "RS")
    gi = check_sensor(gi_acc, 180, fs, "GI")

    taps = np.asarray(bs_taps, dtype=float)

    if taps.ndim != 1 or not np.isfinite(taps).all():
        raise ValueError("Invalid BS tap timestamps")

    if (np.any(taps < 0) or np.any(taps >= 10)
            or np.any(np.diff(taps) <= 0)):
        raise ValueError("Invalid BS tap timing")

    # TS: dynamic acceleration RMS
    ts_magnitude = np.linalg.norm(ts, axis=1)
    ts_dynamic = ts_magnitude - 9.81
    ts_rms = float(np.sqrt(np.mean(ts_dynamic**2)))

    # BS: capacitive touch tapping
    tap_rate = float(len(taps) * 60 / 10)
    tap_std = (
        float(np.std(np.diff(taps)))
        if len(taps) > 1 else None
    )

    # RS: active wrist movement speed proxy
    omega = np.linalg.norm(rs, axis=1)
    peaks, _ = find_peaks(
        omega,
        distance=max(1, int(0.45 * fs)),
        prominence=0.2
    )

    rs_peak = (
        float(np.mean(omega[peaks]))
        if len(peaks) else None
    )

    # GI: candidate steps from 180-second wrist recording
    magnitude = np.linalg.norm(gi, axis=1)

    sos = butter(
        3, [0.6, 3.0],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(sos, magnitude)

    steps, _ = find_peaks(
        filtered,
        distance=max(1, int(0.4 * fs)),
        prominence=0.3
    )

    step_times = steps / fs
    cadence = float(len(steps) * 60 / 180)

    step_std = (
        float(np.std(np.diff(step_times)))
        if len(steps) > 1 else None
    )

    return {
        "TS": {
            "tremor_rms_ms2": ts_rms
        },
        "BS": {
            "tap_count": int(len(taps)),
            "tap_rate_per_min": tap_rate,
            "tap_interval_std_s": tap_std
        },
        "RS": {
            "mean_gyro_peak_rad_s": rs_peak,
            "movement_peaks": int(len(peaks))
        },
        "GI": {
            "step_candidates": int(len(steps)),
            "cadence_steps_min": cadence,
            "step_interval_std_s": step_std
        }
    }


def score_high(value, thresholds):
    return int(sum(value >= x for x in thresholds))


def score_low(value, thresholds):
    return int(sum(value < x for x in thresholds))


def assess_motor(ts_acc, bs_taps, rs_gyro, gi_acc,
                 fs=100, mode="demo"):

    if mode not in ("demo", "real"):
        raise ValueError("Use mode='demo' or mode='real'")

    features = extract_features(
        ts_acc, bs_taps, rs_gyro, gi_acc, fs
    )

    # Real-device readings are NOT clinically scored
    # until thresholds are properly calibrated.
    if mode == "real":
        return {
            "mode": "real",
            "features": features,
            "scores": None,
            "mps": None,
            "status": "UNCALIBRATED - FEATURES ONLY"
        }

    ts = score_high(
        features["TS"]["tremor_rms_ms2"],
        DEMO_THRESHOLDS["TS"]
    )
    bs = score_low(
        features["BS"]["tap_rate_per_min"],
        DEMO_THRESHOLDS["BS"]
    )

    rs_value = features["RS"]["mean_gyro_peak_rad_s"]
    rs = (
        score_low(rs_value, DEMO_THRESHOLDS["RS"])
        if rs_value is not None else None
    )

    gi = score_low(
        features["GI"]["cadence_steps_min"],
        DEMO_THRESHOLDS["GI"]
    )

    scores = {"TS": ts, "BS": bs, "RS": rs, "GI": gi}

    mps = (
        sum(scores.values()) / 16.0
        if all(v is not None for v in scores.values())
        else None
    )

    return {
        "mode": "demo",
        "features": features,
        "scores": scores,
        "mps": mps,
        "status": "DEMO ONLY - NOT CLINICALLY VALIDATED"
    }
