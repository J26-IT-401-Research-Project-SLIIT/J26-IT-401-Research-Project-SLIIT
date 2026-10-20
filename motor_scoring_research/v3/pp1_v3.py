"""PP1 V3: assumption-only simulation. Full operations require Colab CPU.

No learning, label balancing, rejection sampling, or threshold modification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from pathlib import Path

VERSION = "3.0.0"
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "simulation_v3.json"
DEFAULT_ENGINE = HERE.parent / "reference_engine" / "pp1_motor_engine.py"
DOMAINS = ("TS", "BS", "RS", "GI")
FEATURES = {
    "TS": ["tremor_rms_ms2"],
    "BS": ["tap_count", "tap_rate_per_min", "tap_interval_std_s"],
    "RS": ["mean_gyro_peak_rad_s", "movement_peaks"],
    "GI": ["step_candidates", "cadence_steps_min", "step_interval_std_s"],
}
HEADERS = {
    "TS": ["time_s", "ax_ms2", "ay_ms2", "az_ms2"],
    "BS": ["tap_time_s"],
    "RS": ["time_s", "gx_rad_s", "gy_rad_s", "gz_rad_s"],
    "GI": ["time_s", "ax_ms2", "ay_ms2", "az_ms2"],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c["generator_version"] != VERSION or c["sampling_hz"] != 100:
        raise ValueError("Unsupported version or hardware protocol")
    if c["durations_s"] != dict(TS=10, BS=10, RS=10, GI=180):
        raise ValueError("Assessment durations must remain fixed")
    if sum(c["split_counts_per_profile"].values()) != c["sessions_per_profile"]:
        raise ValueError("Split counts do not match session count")
    if set(c["split_counts_per_profile"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split")
    if c["source_policy"] != "assumptions_only_no_empirical_numeric_anchors":
        raise ValueError("Empirical anchors require a separately reviewed source partition design")
    weights = [c[k] for k in ('shared_latent_weight','domain_latent_weight','broad_uniform_weight')]
    if any(w < 0 for w in weights) or abs(sum(weights)-1) > 1e-12:
        raise ValueError("Latent mixture weights must sum to one")
    if c['stress_sessions'] <= 0 or c['stress_sessions'] % 2:
        raise ValueError("Stress count must be positive and even")
    if any(type(n) is not int or n <= 0 for n in c['split_counts_per_profile'].values()):
        raise ValueError("Split counts must be positive integers")
    return c


def require_colab_cpu():
    # Import availability alone is insufficient: also require Colab's /content runtime.
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Full operations are restricted to Google Colab CPU") from exc
    import os
    if not Path("/content").is_dir() or not os.environ.get("COLAB_RELEASE_TAG"):
        raise RuntimeError("A genuine Colab runtime is required")
    if Path("/dev/nvidia0").exists() or os.environ.get("COLAB_TPU_ADDR"):
        raise RuntimeError("Select a CPU runtime (hardware accelerator: None)")


def load_engine(path, config):
    if sha256(path) != config["engine_sha256"]:
        raise ValueError("Original engine hash mismatch; do not edit or silently substitute it")
    spec = importlib.util.spec_from_file_location("pp1_original_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_for(master, identity):
    return int.from_bytes(hashlib.sha256(f"{master}:{identity}".encode()).digest()[:8], "big")


def session_plan(c, stress=False):
    """Metadata-only deterministic assignment, made before signals or scores exist."""
    rows = []
    for profile in ("healthy_like", "symptom_like"):
        count = c["stress_sessions"] // 2 if stress else c["sessions_per_profile"]
        assignments = (["stress"] * count if stress else
                       [s for s, n in c["split_counts_per_profile"].items() for _ in range(n)])
        order = sorted(range(count), key=lambda i: seed_for(c["seed"], f"split:{profile}:{i}"))
        assigned = dict(zip(order, assignments))
        for i in range(count):
            sid = f"V3_{'SHIFT' if stress else 'MAIN'}_{profile}_{i + 1:04d}"
            rows.append({"session_id": sid, "split": assigned[i], "profile": profile,
                         "seed": seed_for(c["seed"], sid),
                         "generator_family": c["stress_family"] if stress else
                         c["families"][seed_for(c["seed"], sid + ':family') % len(c["families"])],
                         "source_reference_id": "SIM_ASSUMPTIONS_V3"})
    return rows


def sample_parameters(c, row):
    import numpy as np
    rng = np.random.default_rng(row["seed"])
    beta = c["healthy_beta"] if row["profile"] == "healthy_like" else c["symptom_beta"]
    shared = float(rng.beta(*beta))
    # Mixture draws broaden coverage without conditioning on engine outputs.
    severity = []
    for _ in DOMAINS:
        choice = rng.random()
        if choice < c["broad_uniform_weight"]:
            severity.append(float(rng.uniform()))
        elif choice < c["broad_uniform_weight"] + c["shared_latent_weight"]:
            severity.append(float(np.clip(shared + rng.normal(0, .10), 0, 1)))
        else:
            severity.append(float(rng.beta(*beta)))
    def draw(key):
        return float(rng.uniform(*c[key]))
    def interpolate(key, s, reverse=False, log=False):
        lo, hi = c[key]
        s = 1 - s if reverse else s
        return float(np.exp(np.log(lo) + s * np.log(hi / lo)) if log else lo + s * (hi-lo))
    p = {"latent_domain_values": severity, "family": row["generator_family"],
         "signal_seed": seed_for(row["seed"], "signals"),
         "ts_amp": interpolate("ts_amplitude_ms2", severity[0], log=True),
         "ts_hz": draw("ts_frequency_hz"), "ts_axis_cos": draw("ts_gravity_axis_cosine"),
         "bs_hz": interpolate("bs_rate_hz", severity[1], reverse=True),
         "bs_cv": draw("bs_interval_cv"),
         "bs_hesitation_probability": draw("bs_hesitation_probability"),
         "bs_fatigue_fraction": float(rng.uniform(0, .25)),
         "rs_amp": interpolate("rs_peak_rad_s", severity[2], reverse=True),
         "rs_hz": draw("rs_frequency_hz"),
         "gi_hz": interpolate("gi_step_frequency_hz", severity[3], reverse=True),
         "gi_amp": draw("gi_dynamic_peak_ms2"),
         "gi_frequency_modulation": draw("gi_frequency_modulation_fraction"),
         "gi_interference": draw("gi_interference_peak_ms2"),
         "acc_noise": draw("acc_noise_std_ms2"), "gyro_noise": draw("gyro_noise_std_rad_s"),
         "orientation_drift": draw("orientation_drift_rad"),
         "rotation_angles_rad": rng.uniform(-np.pi, np.pi, (3, 3)).tolist(),
         "phase_rad": rng.uniform(0, 2*np.pi, 3).tolist(),
         "pause_center_fraction": float(rng.uniform(.25, .75)),
         "pause_duration_s": draw("pause_duration_s")}
    return p


def rotation(angles):
    import numpy as np
    x, y, z = angles
    cx, cy, cz = np.cos(angles); sx, sy, sz = np.sin(angles)
    return (np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @
            np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @
            np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]))


def orient(xyz, t, angles, drift):
    """Rotate gravity and motion together; never add spurious gravity magnitude."""
    import numpy as np
    theta = drift * np.sin(2*np.pi*.13*t)
    out = xyz.copy()
    out[:, 0] = np.cos(theta)*xyz[:, 0] + np.sin(theta)*xyz[:, 2]
    out[:, 2] = -np.sin(theta)*xyz[:, 0] + np.cos(theta)*xyz[:, 2]
    return out @ rotation(angles).T


def tap_events(p, c, rng, duration=10):
    import numpy as np
    taps = []
    now = float(rng.uniform(.08, .4))
    sigma = np.sqrt(np.log1p(p["bs_cv"]**2))
    while now < duration:
        taps.append(now)
        interval = rng.lognormal(-sigma*sigma/2, sigma) / p["bs_hz"]
        interval *= 1 + p["bs_fatigue_fraction"] * now / duration
        if rng.random() < p["bs_hesitation_probability"]:
            interval += rng.uniform(*c["bs_hesitation_s"])
        now += max(c["bs_min_interval_s"], interval)
    return np.asarray(taps)


def _synthesize(p, c, durations):
    """Signal mechanics; short durations are used only by synthetic unit fixtures."""
    import numpy as np
    rng = np.random.default_rng(p["signal_seed"])
    fs = c["sampling_hz"]
    t = np.arange(round(durations['TS']*fs)) / fs
    tg = np.arange(round(durations['GI']*fs)) / fs
    stress = p["family"] == c["stress_family"]
    noise_mult = c["stress_noise_multiplier"] if stress else 1
    intermittent = p["family"] == "intermittent"
    def envelope(time):
        env = np.ones_like(time)
        if p["family"] != "continuous":
            env *= .85 + .15*np.sin(2*np.pi*.23*time)
        if intermittent:
            center = p["pause_center_fraction"] * (time[-1] + 1/fs)
            env *= 1 - .85*np.exp(-.5*((time-center)/(p["pause_duration_s"]/2))**2)
        return env
    def phase(time, hz, offset, variation=0):
        inst = hz*(1 + variation*np.sin(2*np.pi*time/17))
        if stress:
            inst *= 1 + c["stress_chirp_fraction"]*(2*time/(time[-1]+1/fs)-1)
        return offset + 2*np.pi*np.cumsum(inst)/fs
    ts_phase = phase(t, p["ts_hz"], p["phase_rad"][0], .015)
    wave = p["ts_amp"] * envelope(t) * np.sin(ts_phase)
    axis = np.array([np.sqrt(1-p["ts_axis_cos"]**2), 0, p["ts_axis_cos"]])
    ts = wave[:, None]*axis + np.array([0,0,9.81])
    ts[:, 1] += .08*p["ts_amp"]*np.sin(2*ts_phase)
    ts[:, 0] += .025*np.sin(2*np.pi*.35*t)
    rs_phase = phase(t, p["rs_hz"], p["phase_rad"][1], .06)
    rs = np.column_stack([p["rs_amp"]*envelope(t)*np.sin(rs_phase),
                          .07*p["rs_amp"]*np.cos(rs_phase),
                          .04*p["rs_amp"]*np.sin(rs_phase+.3)])
    gi_phase = phase(tg, p["gi_hz"], p["phase_rad"][2], p["gi_frequency_modulation"])
    gw = p["gi_amp"]*envelope(tg)*(1+.08*np.sin(2*np.pi*tg/23))
    gi = np.column_stack([.22*gw*np.sin(gi_phase+.3), .18*gw*np.cos(gi_phase),
                          9.81 + gw*np.sin(gi_phase) + .08*gw*np.sin(2*gi_phase)])
    interference = p["gi_interference"] * (c["stress_interference_multiplier"] if stress else 1)
    gi[:, 2] += interference*np.sin(2*np.pi*1.37*tg + .4)*(.5+.5*np.sin(2*np.pi*tg/31))
    signals = {}
    for i, (d, data, time) in enumerate((("TS",ts,t),("RS",rs,t),("GI",gi,tg))):
        data = orient(data, time, p["rotation_angles_rad"][i], p["orientation_drift"])
        std = p["gyro_noise"] if d == "RS" else p["acc_noise"]
        bias = c["gyro_bias_std_rad_s"] if d == "RS" else c["acc_bias_std_ms2"]
        data += rng.normal(0, std*noise_mult, data.shape) + rng.normal(0, bias, 3)
        signals[d] = data
    signals["BS"] = tap_events(p, c, rng, duration=durations['BS'])
    return signals


def synthesize(p, c):
    """One full protocol session; never permitted in the local runtime."""
    require_colab_cpu()
    return _synthesize(p, c, c['durations_s'])


def verify_preparation(config_path):
    lock = json.loads((HERE/'manifests/preparation_lock.json').read_text())
    for name, digest in lock['files'].items():
        if sha256(HERE/name) != digest:
            raise ValueError(f'Frozen preparation changed: {name}; review and version before running')
    if sha256(config_path) != lock['files']['configs/simulation_v3.json']:
        raise ValueError('Use the frozen V3 config; changes require a separately reviewed version')


def assess(engine, signals, mode="demo"):
    return engine.assess_motor(*(signals[d] for d in DOMAINS), fs=100, mode=mode)


def raw_name(domain, c):
    return f"{domain}_{c['durations_s'][domain]}s.csv"


def save_raw(folder, signals, c):
    import numpy as np
    folder.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for d in DOMAINS:
        data = signals[d]
        if d != "BS":
            data = np.column_stack([np.arange(len(data))/100, data])
        path = folder / raw_name(d, c)
        with path.open("x", encoding="utf-8", newline="") as stream:
            np.savetxt(stream, data, delimiter=",", header=",".join(HEADERS[d]),
                       comments="", fmt="%.17g")
        hashes[d] = sha256(path)
    return hashes


def read_raw(folder, c):
    import numpy as np
    result = {}
    for d in DOMAINS:
        path = folder / raw_name(d, c)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader) != HEADERS[d]:
                raise ValueError(f"Wrong columns: {path}")
            rows = list(reader)
        if any(len(r) != len(HEADERS[d]) for r in rows):
            raise ValueError(f"Malformed row: {path}")
        data = np.array(rows, dtype=float).reshape(-1, len(HEADERS[d]))
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite data: {path}")
        if d == "BS":
            taps = data[:, 0]
            if np.any(taps < 0) or np.any(taps >= 10) or np.any(np.diff(taps) <= 0):
                raise ValueError(f"Invalid tap times: {path}")
            result[d] = taps
        else:
            expected = c["durations_s"][d]*100
            if len(data) != expected or not np.allclose(data[:, 0], np.arange(expected)/100, atol=1e-9, rtol=0):
                raise ValueError(f"Wrong row count/timestamps: {path}")
            result[d] = data[:, 1:]
    return result


def feature_row(sid, features):
    return {"session_id": sid, **{f"{d}_{k}": features[d][k] for d in DOMAINS for k in FEATURES[d]}}


def generate(config_path, engine_path, output, approved=False, stress=False):
    if not approved:
        raise PermissionError("Explicit --approve-generation is required")
    require_colab_cpu()
    import numpy as np
    import scipy
    verify_preparation(config_path)
    c = read_config(config_path)
    engine = load_engine(engine_path, c)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = session_plan(c, stress)
    with (HERE/'manifests'/('stress_plan_v3.csv' if stress else 'session_plan_v3.csv')).open(newline='') as stream:
        frozen_plan = list(csv.DictReader(stream))
    if frozen_plan != [{k: str(v) for k,v in row.items()} for row in plan]:
        raise ValueError('Session plan differs from frozen design')
    write_csv(output / "splits.csv", plan, list(plan[0]))
    write_json(output / "config.json", c)
    write_json(output / "feature_schema.json", FEATURES)
    write_json(output / "source_manifest.json", json.loads((HERE / "manifests/source_manifest.json").read_text()))
    write_json(output / "evaluation_protocol.json", json.loads((HERE / "configs/evaluation_protocol_v3.json").read_text()))
    (output / 'code').mkdir()
    for source in (Path(__file__), Path(engine_path)):
        with (output / 'code' / source.name).open('xb') as stream:
            stream.write(source.read_bytes())
    # Preserve the executable package layout, not just a loose code snapshot.
    lock = json.loads((HERE/'manifests/preparation_lock.json').read_text())
    for name in [*lock['files'], 'manifests/preparation_lock.json']:
        target = output/'preparation'/'v3'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((HERE/name).read_bytes())
    engine_copy = output/'preparation'/'reference_engine'/'pp1_motor_engine.py'
    engine_copy.parent.mkdir(parents=True, exist_ok=True)
    with engine_copy.open('xb') as stream:
        stream.write(Path(engine_path).read_bytes())
    provenance = {"dataset_name": c["dataset_name"], "generator_version": VERSION,
                  "generator_sha256": sha256(__file__), "config_sha256": sha256(output / "config.json"),
                  "engine_sha256": sha256(engine_path), "master_seed": c["seed"],
                  "evaluation": c["independence_claim"], "stress_only": stress,
                  "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
                  "sessions": len(plan), "clinical_labels": False, "MPS_is_probability": False}
    provenance['source_manifest_sha256'] = sha256(output / 'source_manifest.json')
    provenance['evaluation_protocol_sha256'] = sha256(output / 'evaluation_protocol.json')
    provenance['feature_schema_sha256'] = sha256(output / 'feature_schema.json')
    provenance['splits_sha256'] = sha256(output / 'splits.csv')
    provenance['preparation_lock_sha256'] = sha256(HERE/'manifests/preparation_lock.json')
    write_json(output / "dataset_manifest.json", provenance)
    features, targets = [], []
    with (output / "sessions.jsonl").open("x", encoding="utf-8") as manifest:
        for row in plan:
            p = sample_parameters(c, row)
            signals = synthesize(p, c)
            hashes = save_raw(output / "cases" / row["session_id"], signals, c)
            result = assess(engine, signals)
            real = assess(engine, signals, "real")
            if real["scores"] is not None or real["mps"] is not None:
                raise ValueError("Real mode must remain features-only")
            features.append(feature_row(row["session_id"], result["features"]))
            targets.append({"session_id": row["session_id"], **result["scores"], "demo_MPS": result["mps"]})
            manifest.write(json.dumps({**row, **provenance, "parameters": p, "raw_sha256": hashes,
                                       "scores": result["scores"], "demo_MPS": result["mps"]}, allow_nan=False) + "\n")
    write_csv(output / "features.csv", features, list(features[0]))
    write_csv(output / "targets.csv", targets, list(targets[0]))
    # A failed/interrupted run remains visible and is never resumed or overwritten.
    report = audit(output, engine_path)
    write_json(output / "integrity_report.json", report)
    write_json(output / "COMPLETE.json", {"integrity_passed": True, "sessions": len(plan)})
    return report


def audit(dataset, engine_path):
    require_colab_cpu()
    import numpy as np
    root = Path(dataset); c = read_config(root / "config.json")
    engine = load_engine(engine_path, c)
    meta = json.loads((root / "dataset_manifest.json").read_text())
    if meta["config_sha256"] != sha256(root / "config.json") or meta["engine_sha256"] != sha256(engine_path):
        raise ValueError("Configuration/engine provenance changed")
    for key, file in [('source_manifest_sha256','source_manifest.json'),
                      ('evaluation_protocol_sha256','evaluation_protocol.json'),
                      ('feature_schema_sha256','feature_schema.json'),('splits_sha256','splits.csv'),
                      ('generator_sha256','code/pp1_v3.py'),('engine_sha256','code/pp1_motor_engine.py'),
                      ('preparation_lock_sha256','preparation/v3/manifests/preparation_lock.json')]:
        if meta[key] != sha256(root/file):
            raise ValueError(f'Artifact changed: {file}')
    package = root/'preparation'/'v3'
    package_lock = json.loads((package/'manifests/preparation_lock.json').read_text())
    for name, digest in package_lock['files'].items():
        if sha256(package/name) != digest:
            raise ValueError(f'Preserved preparation changed: {name}')
    if sha256(root/'preparation/reference_engine/pp1_motor_engine.py') != meta['engine_sha256']:
        raise ValueError('Preserved engine changed')
    if json.loads((root/'feature_schema.json').read_text()) != FEATURES:
        raise ValueError('Unexpected feature schema')
    def table(name):
        with (root/name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len({r['session_id'] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        if name == 'features.csv' and (not rows or set(rows[0]) !=
                {'session_id'} | {f'{d}_{k}' for d in DOMAINS for k in FEATURES[d]}):
            raise ValueError('Feature columns violate allowlist')
        return {r['session_id']: r for r in rows}
    splits, targets, features = (table(n) for n in ("splits.csv", "targets.csv", "features.csv"))
    records = [json.loads(line) for line in (root/"sessions.jsonl").read_text().splitlines()]
    expected_plan = {r['session_id']: r for r in session_plan(c, meta["stress_only"])}
    if not (set(splits) == set(targets) == set(features) == set(expected_plan) ==
            {r['session_id'] for r in records} == {x.name for x in (root/'cases').iterdir()}):
        raise ValueError("Session membership mismatch")
    if len(records) != len(expected_plan) or len(records) != meta['sessions']:
        raise ValueError("Session count mismatch")
    seen = {d: {} for d in DOMAINS}; counts = {}; null_scores = Counter(); complete = 0
    for row in records:
        sid = row['session_id']; split = row['split']
        for key, value in expected_plan[sid].items():
            if str(row[key]) != str(value) or splits[sid][key] != str(value):
                raise ValueError(f"Changed split/provenance: {sid}:{key}")
        for key in ("generator_sha256", "engine_sha256", "config_sha256", "generator_version"):
            if row[key] != meta[key]:
                raise ValueError(f"Inconsistent provenance: {sid}:{key}")
        signals = read_raw(root/'cases'/sid, c)
        if {p.name for p in (root/'cases'/sid).iterdir()} != {raw_name(d,c) for d in DOMAINS}:
            raise ValueError(f'Unexpected files: {sid}')
        if row['parameters'] != sample_parameters(c, expected_plan[sid]):
            raise ValueError(f'Nonreproducible parameter manifest: {sid}')
        for d in DOMAINS:
            digest = sha256(root/'cases'/sid/raw_name(d,c))
            if digest != row['raw_sha256'][d]:
                raise ValueError(f"Hash mismatch: {sid}:{d}")
            if digest in seen[d]:
                raise ValueError(f"Duplicate raw recording: {sid}, {seen[d][digest]}, {d}")
            seen[d][digest] = sid
        result = assess(engine, signals)
        if result['scores'] != row['scores'] or result['mps'] != row['demo_MPS']:
            raise ValueError(f"Serialized signal scoring mismatch: {sid}")
        real = assess(engine, signals, 'real')
        if real['scores'] is not None or real['mps'] is not None:
            raise ValueError("Real-device scoring regression")
        for key, value in feature_row(sid, result['features']).items():
            if key == 'session_id':
                continue
            saved = features[sid][key]
            if (value is None and saved != '') or (value is not None and
                    (saved == '' or not np.isclose(float(saved), value, atol=1e-10, rtol=1e-10))):
                raise ValueError(f"Feature mismatch: {sid}:{key}")
        for d in DOMAINS:
            value = result['scores'][d]
            if targets[sid][d] != ('' if value is None else str(value)):
                raise ValueError(f"Target mismatch: {sid}:{d}")
            counts.setdefault(split, {domain: [0]*5 for domain in DOMAINS})
            if value is None:
                null_scores[d] += 1
            elif type(value) is not int or value not in range(5):
                raise ValueError("Invalid ordinal score")
            else:
                counts[split][d][value] += 1
        if result['mps'] is not None:
            if result['mps'] != sum(result['scores'].values())/16 or float(targets[sid]['demo_MPS']) != result['mps']:
                raise ValueError("MPS mismatch")
            complete += 1
        elif targets[sid]['demo_MPS'] != '':
            raise ValueError("Invented MPS for incomplete scores")
    warnings = [f'{split}/{d}/score{k}: only {n} cases' for split, domains in counts.items()
                for d, values in domains.items() for k, n in enumerate(values) if n < 5]
    return {"status": "passed", "session_count": len(records), "complete_score_sessions": complete,
            "complete_score_coverage": complete/len(records), "class_counts_0_to_4": counts,
            "null_scores": dict(null_scores), "warnings": warnings,
            "source_independence": False, "family_independence": False,
            "baseline": "Unchanged engine reproduces its own labels; identity check, not clinical validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'audit'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approve-generation', action='store_true')
    parser.add_argument('--stress', action='store_true')
    args = parser.parse_args()
    result = (generate(args.config, args.engine, args.output, args.approve_generation, args.stress)
              if args.command == 'generate' else audit(args.output, args.engine))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
