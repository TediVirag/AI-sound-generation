"""
Build an AI learning dataset from PostgreSQL table `sound_samples` reachable via SSH tunnel.

Schema:
- sound_samples.json      : TEXT, JSON string in concatenation format (multiple sounds per row)
- sound_samples.emotion   : VARCHAR, one of: surprise, disgust, fear, anger, sad, happy
- sound_samples.strength  : VARCHAR, whole number 1-5 (1=weakest, 5=strongest)

Dataset goal (numeric encoding):
- X : emotion (one-hot, 6-dim) + strength (scalar normalized, 1-dim) = 7-dim vector
- Y : fixed-size numeric encoding of the JSON structure

Compatible with audio_synthesis_tool3.py.

What is encoded per sound (up to MAX_SOUNDS):
  Concat  : target_sample_rate, intervals (MAX_SOUNDS-1 slots)
  Global  : sample_rate, duration, num_partials, num_formants,
            vocal_style (one-hot, 4-dim), inharmonicity,
            global_sweep_rate, global_trill_rate, noise_level     -> 12 values
  Partials: per partial (up to MAX_PARTIALS):
            freq, amp, attack, decay, vibrato, vib_rate,
            vib_depth, inharmonic, distortion, sweep_rate,
            envelope_type (one-hot, 3-dim)                        -> 13 values each
  Formants: per formant (up to MAX_FORMANTS):
            freq, bandwidth                                        -> 2 values each
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import psycopg2
import psycopg2.extras


# ---------------------------
# CONFIG — EDIT THESE VALUES
# ---------------------------

@dataclass
class Config:
    # SSH jump host (bastion)
    ssh_host: str = "caesar.elte.hu"
    ssh_port: int = 22
    ssh_username: str = "tedivirag"

    # Auth: use key (recommended) or password
    ssh_pkey: Optional[str] = "/path/to/id_rsa"
    ssh_password: Optional[str] = "0107LuckyV+"

    # DB host/port as reachable FROM the bastion
    remote_db_host: str = "pgsql.caesar.elte.hu"
    remote_db_port: int = 5432

    # PostgreSQL credentials
    db_name: str = "p-soundgen"
    db_user: str = "tedivirag"
    db_password: str = "0107LuckyV+"
    db_sslmode: str = "prefer"

    # Table / column names
    table: str = "aggregated_results_1"
    col_json: str = "json"
    col_emotion: str = "emotion"
    col_strength: str = "strength"

    # Optional row limit for debugging (None = fetch all)
    limit: Optional[int] = None

    # Output folder
    out_dir: str = "dataset_out"

    # Train/val split ratio
    train_split: float = 0.9
    seed: int = 42


# ---------------------------
# Encoding constants
# Do NOT change after generating a dataset
# ---------------------------

ALLOWED_EMOTIONS  = ["surprise", "disgust", "fear", "anger", "sad", "happy"]
EMOTION_TO_ID     = {e: i for i, e in enumerate(ALLOWED_EMOTIONS)}

ALLOWED_STRENGTHS = {1, 2, 3, 4, 5}
STRENGTH_MIN, STRENGTH_MAX = 1, 5

# Must match VOCAL_PRESETS keys in audio_synthesis_tool3.py
VOCAL_STYLES      = ["melodic", "percussive", "breathy", "harmonic"]
VOCAL_STYLE_TO_ID = {s: i for i, s in enumerate(VOCAL_STYLES)}

# Must match envelope types used in audio_synthesis_tool3.py
ENVELOPE_TYPES    = ["exponential", "sharp_attack", "smooth_swell"]
ENVELOPE_TO_ID    = {e: i for i, e in enumerate(ENVELOPE_TYPES)}

MAX_SOUNDS    = 4
MAX_PARTIALS  = 6
MAX_FORMANTS  = 4

# Normalisation ranges
FREQ_MAX          = 20000.0
AMP_MAX           = 1.0
ATTACK_MAX        = 2000.0   # ms
DECAY_MAX         = 2000.0   # ms
VIB_RATE_MAX      = 50.0
VIB_DEPTH_MAX     = 200.0
DISTORTION_MAX    = 5.0
SWEEP_RATE_MAX    = 5.0
DURATION_MAX      = 5000.0   # ms
SAMPLE_RATE_MAX   = 48000.0
INHARMONICITY_MAX = 1.0
GLOBAL_SWEEP_MAX  = 5.0
GLOBAL_TRILL_MAX  = 50.0
NOISE_LEVEL_MAX   = 1.0
INTERVAL_MAX      = 5000.0   # ms


# ---------------------------
# Normalisation helpers
# ---------------------------

def _clip_norm(value: float, max_val: float) -> float:
    if max_val == 0:
        return 0.0
    return float(np.clip(value / max_val, 0.0, 1.0))


def _denorm(value: float, max_val: float) -> float:
    return float(np.clip(value, 0.0, 1.0) * max_val)


def _one_hot(idx: int, size: int) -> List[float]:
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _argmax(values: List[float]) -> int:
    return int(np.argmax(values))


# ---------------------------
# Encoder: JSON dict -> flat float32 vector
# ---------------------------

def encode_json(sample_json: Dict[str, Any]) -> np.ndarray:
    """
    Encode the full concatenation JSON into a fixed-size float32 vector.
    Encodes DB values faithfully. Unused slots are zero-padded.
    """
    vector: List[float] = []

    concat      = sample_json.get("concatenation", {}) or {}
    sound_files = concat.get("sound_files", []) or []
    intervals   = concat.get("intervals", []) or []
    sounds      = [s for s in sound_files if s is not None]

    vector.append(_clip_norm(float(concat.get("sample_rate", 44100)), SAMPLE_RATE_MAX))
    for i in range(MAX_SOUNDS - 1):
        val = float(intervals[i]) if i < len(intervals) else 0.0
        vector.append(_clip_norm(max(0.0, val), INTERVAL_MAX))

    for s_idx in range(MAX_SOUNDS):
        if s_idx < len(sounds):
            sound    = sounds[s_idx]
            g        = sound.get("global", {}) or {}
            partials = sound.get("partials", []) or []
            formants = sound.get("formants", []) or []

            vector.append(_clip_norm(float(g.get("sample_rate",      44100)), SAMPLE_RATE_MAX))
            vector.append(_clip_norm(float(g.get("duration",           300)), DURATION_MAX))
            vector.append(float(g.get("num_partials", 1)) / MAX_PARTIALS)
            vector.append(float(g.get("num_formants", 0)) / MAX_FORMANTS)
            vs_id = VOCAL_STYLE_TO_ID.get(str(g.get("vocal_style", "")), 0)
            vector.extend(_one_hot(vs_id, len(VOCAL_STYLES)))
            vector.append(_clip_norm(float(g.get("inharmonicity",      0.1)), INHARMONICITY_MAX))
            vector.append(_clip_norm(float(g.get("global_sweep_rate",  0.5)), GLOBAL_SWEEP_MAX))
            vector.append(_clip_norm(float(g.get("global_trill_rate",   12)), GLOBAL_TRILL_MAX))
            vector.append(_clip_norm(float(g.get("noise_level",        0.02)), NOISE_LEVEL_MAX))

            for p_idx in range(MAX_PARTIALS):
                if p_idx < len(partials):
                    p      = partials[p_idx]
                    env_id = ENVELOPE_TO_ID.get(str(p.get("envelope_type", "exponential")), 0)
                    vector.append(_clip_norm(float(p.get("freq",       440)), FREQ_MAX))
                    vector.append(_clip_norm(float(p.get("amp",        0.5)), AMP_MAX))
                    vector.append(_clip_norm(float(p.get("attack",      50)), ATTACK_MAX))
                    vector.append(_clip_norm(float(p.get("decay",      200)), DECAY_MAX))
                    vector.append(1.0 if bool(p.get("vibrato")) else 0.0)
                    vector.append(_clip_norm(float(p.get("vib_rate",    5)), VIB_RATE_MAX))
                    vector.append(_clip_norm(float(p.get("vib_depth",  20)), VIB_DEPTH_MAX))
                    vector.append(1.0 if bool(p.get("inharmonic")) else 0.0)
                    vector.append(_clip_norm(float(p.get("distortion",  0)), DISTORTION_MAX))
                    vector.append(_clip_norm(float(p.get("sweep_rate",  0)), SWEEP_RATE_MAX))
                    vector.extend(_one_hot(env_id, len(ENVELOPE_TYPES)))
                else:
                    vector.extend([0.0] * 13)

            for f_idx in range(MAX_FORMANTS):
                if f_idx < len(formants):
                    f = formants[f_idx]
                    vector.append(_clip_norm(float(f.get("freq",      800)), FREQ_MAX))
                    vector.append(_clip_norm(float(f.get("bandwidth", 200)), FREQ_MAX))
                else:
                    vector.extend([0.0] * 2)
        else:
            vector.extend([0.0] * (12 + MAX_PARTIALS * 13 + MAX_FORMANTS * 2))

    return np.array(vector, dtype=np.float32)

# Precomputed — used to validate loaded .npy files
VECTOR_SIZE = (
    1 + (MAX_SOUNDS - 1)           # concat-level: sample_rate + intervals
    + MAX_SOUNDS * (
        12                         # global per sound
        + MAX_PARTIALS * 13        # partials
        + MAX_FORMANTS * 2         # formants
    )
)

# ---------------------------
# Decoder: flat float32 vector -> JSON dict
# Compatible with audio_synthesis_tool3.py
# ---------------------------

def decode_vector(vector: np.ndarray) -> Dict[str, Any]:
    """
    Decode a float32 vector back into the concatenation JSON structure.
    Filters out 'ghost sounds' generated by unpenalized neural network biases 
    by checking unclamped values before applying bounds.
    """
    v   = list(vector)
    idx = 0

    def take(n: int) -> List[float]:
        nonlocal idx
        chunk = v[idx: idx + n]
        idx  += n
        return chunk

    def take1() -> float:
        return take(1)[0]

    target_sr = max(8000, round(_denorm(take1(), SAMPLE_RATE_MAX)))
    intervals = [max(0, round(_denorm(x, INTERVAL_MAX))) for x in take(MAX_SOUNDS - 1)]

    sound_files = []

    for _ in range(MAX_SOUNDS):
        # 1. Read all raw network outputs for the sound FIRST to keep the index pointer aligned
        raw_sr        = take1()
        raw_dur       = take1()
        raw_n_part    = take1()
        raw_n_form    = take1()
        vs_oh         = take(len(VOCAL_STYLES))
        raw_inhar     = take1()
        raw_gsweep    = take1()
        raw_gtrill    = take1()
        raw_noise     = take1()

        raw_partials = []
        for _ in range(MAX_PARTIALS):
            raw_freq   = take1()
            raw_amp    = take1()
            raw_att    = take1()
            raw_dec    = take1()
            raw_vib    = take1()
            raw_vibrat = take1()
            raw_vibdep = take1()
            raw_inharm = take1()
            raw_dist   = take1()
            raw_sweep  = take1()
            env_oh     = take(len(ENVELOPE_TYPES))
            raw_partials.append((
                raw_freq, raw_amp, raw_att, raw_dec, raw_vib,
                raw_vibrat, raw_vibdep, raw_inharm, raw_dist,
                raw_sweep, env_oh
            ))

        raw_formants = []
        for _ in range(MAX_FORMANTS):
            f_freq = take1()
            f_bw   = take1()
            raw_formants.append((f_freq, f_bw))

        # 2. Check if this sound is actually just empty padding/bias noise
        dur_unclamped = round(_denorm(raw_dur, DURATION_MAX))
        if dur_unclamped < 5:  # If duration is effectively 0, skip keeping it
            continue

        # 3. Apply normal bounds and extract data since the sound is valid
        sample_rate   = max(8000, round(_denorm(raw_sr, SAMPLE_RATE_MAX)))
        duration      = max(10,   dur_unclamped)
        num_partials  = min(MAX_PARTIALS, max(1, round(raw_n_part * MAX_PARTIALS)))
        num_formants  = min(MAX_FORMANTS, max(0, round(raw_n_form * MAX_FORMANTS)))
        
        vocal_style   = VOCAL_STYLES[_argmax(vs_oh)]
        inharmonicity = max(0.0, round(_denorm(raw_inhar, INHARMONICITY_MAX), 4))
        global_sweep  = max(0.0, round(_denorm(raw_gsweep, GLOBAL_SWEEP_MAX),  4))
        global_trill  = max(0.0, round(_denorm(raw_gtrill, GLOBAL_TRILL_MAX),  4))
        noise_level   = max(0.0, round(_denorm(raw_noise, NOISE_LEVEL_MAX),   4))

        partials = []
        for (r_freq, r_amp, r_att, r_dec, r_vib,
             r_vibrat, r_vibdep, r_inharm, r_dist,
             r_sweep, env_oh) in raw_partials[:num_partials]:
            
            freq_unclamped = round(_denorm(r_freq, FREQ_MAX), 2)
            amp_unclamped  = round(_denorm(r_amp, AMP_MAX), 4)
            
            # Skip partials that are practically zero (ghost partials)
            if freq_unclamped < 1.0 or amp_unclamped < 0.001:
                continue
                
            freq       = max(20.0, freq_unclamped)
            amp        = max(0.01, amp_unclamped)
            attack     = max(1.0,  round(_denorm(r_att, ATTACK_MAX), 1))
            decay      = max(1.0,  round(_denorm(r_dec, DECAY_MAX),  1))
            vibrato    = r_vib >= 0.5
            vib_rate   = max(0.0,  round(_denorm(r_vibrat, VIB_RATE_MAX),  2))
            vib_depth  = max(0.0,  round(_denorm(r_vibdep, VIB_DEPTH_MAX), 2))
            inharmonic = r_inharm >= 0.5
            distortion = max(0.0,  round(_denorm(r_dist, DISTORTION_MAX), 4))
            sweep_rate = max(0.0,  round(_denorm(r_sweep, SWEEP_RATE_MAX), 4))
            env_type   = ENVELOPE_TYPES[_argmax(env_oh)]
            
            partials.append({
                "freq":          freq,
                "amp":           amp,
                "attack":        attack,
                "decay":         decay,
                "vibrato":       vibrato,
                "vib_rate":      vib_rate  if vibrato else 0,
                "vib_depth":     vib_depth if vibrato else 0,
                "inharmonic":    inharmonic,
                "distortion":    distortion,
                "sweep_rate":    sweep_rate,
                "envelope_type": env_type,
            })

        formants = []
        for (r_freq, r_bw) in raw_formants[:num_formants]:
            freq_unclamped = round(_denorm(r_freq, FREQ_MAX), 2)
            if freq_unclamped < 1.0:
                continue
                
            freq      = max(20.0, freq_unclamped)
            bandwidth = max(10.0, round(_denorm(r_bw, FREQ_MAX), 2))
            formants.append({"freq": freq, "bandwidth": bandwidth})

        # If all partials were ghost partials and got filtered out, drop the sound entirely
        if not partials:
            continue

        sound_files.append({
            "global": {
                "sample_rate":       sample_rate,
                "duration":          duration,
                "num_partials":      len(partials),
                "num_formants":      len(formants),
                "vocal_style":       vocal_style,
                "inharmonicity":     inharmonicity,
                "global_sweep_rate": global_sweep,
                "global_trill_rate": global_trill,
                "noise_level":       noise_level,
            },
            "partials": partials,
            "formants": formants,
        })

    n_sounds  = len(sound_files)
    intervals = intervals[:max(0, n_sounds - 1)]
    order     = list(range(n_sounds))

    return {
        "concatenation": {
            "sound_files": sound_files,
            "intervals":   intervals,
            "order":       order,
            "sample_rate": target_sr,
        }
    }
# ---------------------------
# Label / strength helpers
# ---------------------------

def parse_strength(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return None
    return value if value in ALLOWED_STRENGTHS else None


def encode_x(emotion: str, strength: int) -> np.ndarray:
    """7-dim float32 X vector: emotion one-hot (6) + strength norm (1)."""
    emotion_vec  = _one_hot(EMOTION_TO_ID[emotion], len(ALLOWED_EMOTIONS))
    strength_val = (strength - STRENGTH_MIN) / (STRENGTH_MAX - STRENGTH_MIN)
    return np.array(emotion_vec + [strength_val], dtype=np.float32)


# ---------------------------
# DB access via SSH tunnel
# ---------------------------

def open_ssh_tunnel(cfg: Config) -> SSHTunnelForwarder:
    kwargs: Dict[str, Any] = {
        "ssh_address_or_host": (cfg.ssh_host, cfg.ssh_port),
        "ssh_username":        cfg.ssh_username,
        "remote_bind_address": (cfg.remote_db_host, cfg.remote_db_port),
    }
    if cfg.ssh_pkey:
        kwargs["ssh_pkey"] = cfg.ssh_pkey
    if cfg.ssh_password:
        kwargs["ssh_password"] = cfg.ssh_password
    server = SSHTunnelForwarder(**kwargs)
    server.start()
    return server


def fetch_rows(cfg: Config, local_port: int) -> List[Dict[str, Any]]:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=local_port,
        dbname=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
        sslmode=cfg.db_sslmode,
    )
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = f"""
                SELECT
                    s.{cfg.col_json}       AS json_text,
                    res.{cfg.col_emotion}  AS emotion,
                    res.{cfg.col_strength} AS strength
                FROM {cfg.table} res JOIN sound_samples s ON res.sound_code = s.sound_code
            """
            if cfg.limit is not None:
                sql += " LIMIT %s"
                cur.execute(sql, (cfg.limit,))
            else:
                cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------
# Dataset construction
# ---------------------------

def build_dataset(
    records: List[Dict[str, Any]],
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid: List[Dict[str, Any]] = []
    skipped = {"bad_emotion": 0, "bad_strength": 0, "bad_json": 0}

    for r in records:
        emotion = (r.get("emotion") or "").strip().lower()
        if emotion not in EMOTION_TO_ID:
            skipped["bad_emotion"] += 1
            continue

        strength = parse_strength(r.get("strength"))
        if strength is None:
            skipped["bad_strength"] += 1
            continue

        try:
            raw         = r.get("json_text") or ""
            sample_json = json.loads(raw.strip() if isinstance(raw, str) else json.dumps(raw))
            y_vector    = encode_json(sample_json)
        except Exception:
            skipped["bad_json"] += 1
            continue

        valid.append({
            "x": encode_x(emotion, strength),
            "y": y_vector,
        })

    print(
        "Filtering summary:\n"
        f"  total fetched : {len(records)}\n"
        f"  kept          : {len(valid)}\n"
        f"  bad_emotion   : {skipped['bad_emotion']}\n"
        f"  bad_strength  : {skipped['bad_strength']}\n"
        f"  bad_json      : {skipped['bad_json']}"
    )

    rng = random.Random(cfg.seed)
    rng.shuffle(valid)
    split_idx = int(len(valid) * cfg.train_split)
    return valid[:split_idx], valid[split_idx:]


def save_outputs(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    cfg: Config,
) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    for records, split in [(train, "train"), (val, "val")]:
        x_arr = np.stack([r["x"] for r in records])
        y_arr = np.stack([r["y"] for r in records])
        np.save(os.path.join(cfg.out_dir, f"{split}_x.npy"), x_arr)
        np.save(os.path.join(cfg.out_dir, f"{split}_y.npy"), y_arr)
        print(f"  {split}_x.npy  {x_arr.shape}")
        print(f"  {split}_y.npy  {y_arr.shape}")

    meta = {
        "x_size":           7,
        "y_size":           VECTOR_SIZE,
        "emotion_to_id":    EMOTION_TO_ID,
        "allowed_emotions": ALLOWED_EMOTIONS,
        "strength_values":  sorted(ALLOWED_STRENGTHS),
        "vocal_styles":     VOCAL_STYLES,
        "envelope_types":   ENVELOPE_TYPES,
        "max_sounds":       MAX_SOUNDS,
        "max_partials":     MAX_PARTIALS,
        "max_formants":     MAX_FORMANTS,
        "n_train":          len(train),
        "n_val":            len(val),
    }
    meta_path = os.path.join(cfg.out_dir, "label_maps.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  {meta_path}")


def main() -> None:
    cfg = Config()

    missing = [
        name for name, val in [
            ("ssh_host",     cfg.ssh_host),
            ("ssh_username", cfg.ssh_username),
            ("db_name",      cfg.db_name),
            ("db_user",      cfg.db_user),
            ("db_password",  cfg.db_password),
        ]
        if not val or val.startswith("YOUR_")
    ]
    if missing:
        raise SystemExit(f"Please fill in Config values: {', '.join(missing)}")

    tunnel = open_ssh_tunnel(cfg)
    try:
        print(f"SSH tunnel: 127.0.0.1:{tunnel.local_bind_port} -> {cfg.remote_db_host}:{cfg.remote_db_port}")
        records = fetch_rows(cfg, tunnel.local_bind_port)
        print(f"Fetched {len(records)} rows from '{cfg.table}'")
        train, val = build_dataset(records, cfg)
        save_outputs(train, val, cfg)
    finally:
        tunnel.stop()


if __name__ == "__main__":
    main()