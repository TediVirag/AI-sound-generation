"""
Train a feedforward neural network on the SoundSamplesDataset.

Key improvements over v1:
  - FiLM-conditioned MLP: emotion one-hot directly modulates every hidden layer
    via learned scale+shift, making emotion the dominant signal.
  - MSE + TripletMarginLoss: contrastive loss explicitly pushes different-emotion
    outputs apart in the embedding space, preventing regression-to-the-mean.
  - Larger default capacity and longer patience.

Input  X : float32 (7,)   — [emotion_one_hot (6) | strength_norm (1)]
Output Y : float32 (VECTOR_SIZE,) — encoded JSON vector

Usage:
    python train_sound_mlp.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from sound_samples_dataset import SoundSamplesDataset, collate_fn, DATASET_DIR
from dataset_gen import (
    decode_vector,
    ALLOWED_EMOTIONS,
    ALLOWED_STRENGTHS,
    STRENGTH_MIN,
    STRENGTH_MAX,
    VECTOR_SIZE,
    EMOTION_TO_ID,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE       = 64
NUM_WORKERS      = 0
EPOCHS           = 200          # more room to converge
LEARNING_RATE    = 3e-4         # lower LR → smoother loss curve
WEIGHT_DECAY     = 1e-5
VALIDATION_SPLIT = 0.15         # larger val set → more stable val signal

PATIENCE         = 30           # don't stop on noise

HIDDEN_DIMS      = [512, 1024, 1024, 512]
DROPOUT          = 0.15

# ── Loss weights ──────────────────────────────────────────────────────────────
# Start triplet weight LOW and ramp it up after MSE is stable.
# Phase 1 (epochs 1-30):  pure MSE — learn the basic reconstruction first
# Phase 2 (epochs 31+):   add triplet gradually
MSE_WEIGHT      = 1.0
TRIPLET_WEIGHT  = 0.15          # reduced from 0.4  — stop it dominating
TRIPLET_MARGIN  = 0.3           # reduced from 0.5  — easier target early on
TRIPLET_WARMUP  = 30            # epochs of MSE-only before triplet is enabled

CHECKPOINT_PATH   = "best_sound_model.pt"
SAMPLE_OUTPUT_DIR = "generated_samples"
N_SAMPLE_OUTPUTS  = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# FiLM-conditioned MLP
# ─────────────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Learns scale (γ) and shift (β) from the condition vector c,
    then applies: out = γ(c) * x + β(c)
    This makes the condition (emotion) actively reshape every hidden layer.
    """
    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta  = nn.Linear(condition_dim, feature_dim)
        # Init gamma near 1 so early training is stable
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.gamma(c) * x + self.beta(c)


class SoundMLP(nn.Module):
    """
    FiLM-conditioned MLP.

    The 7-dim input is split:
      - emotion_emb  (6-dim one-hot) → condition signal fed to every FiLM layer
      - strength     (1-dim)         → concatenated at input

    Architecture per block:
        Linear → LayerNorm → GELU → FiLM(emotion) → Dropout
    """

    def __init__(
        self,
        input_dim:     int = 7,
        output_dim:    int = VECTOR_SIZE,
        hidden_dims:   List[int] = None,
        dropout:       float = DROPOUT,
        condition_dim: int = 6,       # emotion one-hot size
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        self.condition_dim = condition_dim

        # Project condition (emotion one-hot) to a richer embedding
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        cond_size = 64

        self.linears   = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.films     = nn.ModuleList()
        self.drops     = nn.ModuleList()

        prev = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(prev, h))
            self.norms.append(nn.LayerNorm(h))
            self.films.append(FiLMLayer(h, cond_size))
            self.drops.append(nn.Dropout(dropout))
            prev = h

        self.head = nn.Linear(prev, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split condition from input
        emotion = x[:, :self.condition_dim]          # (B, 6)
        c = self.condition_proj(emotion)              # (B, 64)

        h = x
        for linear, norm, film, drop in zip(
            self.linears, self.norms, self.films, self.drops
        ):
            h = linear(h)
            h = norm(h)
            h = F.gelu(h)
            h = film(h, c)                           # emotion modulates every layer
            h = drop(h)

        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Triplet dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Wraps SoundSamplesDataset and yields (anchor, positive, negative) triplets.

    anchor   — a sample with emotion E
    positive — another sample with the SAME emotion E
    negative — a sample with a DIFFERENT emotion
    """

    def __init__(self, base: SoundSamplesDataset) -> None:
        self.base = base
        # Build per-emotion index lists (into base.indices)
        self.by_emotion: Dict[int, List[int]] = {}
        emotion_ids = base.x[base.indices, :6].argmax(dim=1).tolist()
        for local_i, eid in enumerate(emotion_ids):
            self.by_emotion.setdefault(eid, []).append(local_i)
        self.all_emotions = list(self.by_emotion.keys())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_sample = self.base[idx]
        anchor_eid    = int(anchor_sample["x"][:6].argmax().item())

        # Positive: same emotion, different index
        pos_pool = self.by_emotion[anchor_eid]
        pos_idx  = idx
        if len(pos_pool) > 1:
            while pos_idx == idx:
                pos_idx = random.choice(pos_pool)
        pos_sample = self.base[pos_idx]

        # Negative: different emotion
        neg_emotions = [e for e in self.all_emotions if e != anchor_eid]
        neg_eid      = random.choice(neg_emotions)
        neg_idx      = random.choice(self.by_emotion[neg_eid])
        neg_sample   = self.base[neg_idx]

        return {
            "x":   anchor_sample["x"],
            "y":   anchor_sample["y"],
            "y_p": pos_sample["y"],
            "y_n": neg_sample["y"],
        }


def triplet_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "x":   torch.stack([s["x"]   for s in batch]),
        "y":   torch.stack([s["y"]   for s in batch]),
        "y_p": torch.stack([s["y_p"] for s in batch]),
        "y_n": torch.stack([s["y_n"] for s in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    mse_crit:     nn.Module,
    trip_crit:    nn.Module,
    optimizer:    torch.optim.Optimizer | None,
    device:       torch.device,
    train:        bool,
    triplet_on:   bool = True,       # ← NEW: disable triplet during warmup
) -> Tuple[float, float, float]:
    model.train(train)
    total_mse, total_trip, total = 0.0, 0.0, 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x   = batch["x"].to(device)
            y   = batch["y"].to(device)
            y_p = batch["y_p"].to(device)
            y_n = batch["y_n"].to(device)

            pred = model(x)
            mse  = mse_crit(pred, y)

            if triplet_on:
                trip = trip_crit(pred, y_p, y_n)
                loss = MSE_WEIGHT * mse + TRIPLET_WEIGHT * trip
            else:
                trip = torch.tensor(0.0)
                loss = mse                           # pure MSE during warmup

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # tighter clip
                optimizer.step()

            total_mse  += mse.item()
            total_trip += trip.item()
            total      += loss.item()
            n          += 1

    n = max(n, 1)
    return total / n, total_mse / n, total_trip / n


def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
) -> None:
    mse_crit  = nn.MSELoss()
    trip_crit = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Warmup LR for first 10 epochs, then cosine decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - 10, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[10],
    )

    best_val, no_improv = float("inf"), 0

    print(f"\nTraining on {device}  |  epochs={EPOCHS}  |  patience={PATIENCE}")
    print(f"Warmup (MSE only): epochs 1–{TRIPLET_WARMUP}")
    print(f"After warmup: {MSE_WEIGHT}×MSE + {TRIPLET_WEIGHT}×Triplet(margin={TRIPLET_MARGIN})\n")
    print(f"{'Epoch':>5}  {'Total':>10}  {'MSE':>10}  {'Triplet':>10}  {'Val':>10}  {'LR':>9}  Phase")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        triplet_on = (epoch > TRIPLET_WARMUP)
        phase      = "MSE+Trip" if triplet_on else "MSE only"

        tr_total, tr_mse, tr_trip = run_epoch(
            model, train_loader, mse_crit, trip_crit,
            optimizer, device, train=True, triplet_on=triplet_on,
        )
        val_total, val_mse, val_trip = run_epoch(
            model, val_loader, mse_crit, trip_crit,
            None, device, train=False, triplet_on=triplet_on,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>5}  {tr_total:>10.5f}  {tr_mse:>10.5f}  "
            f"{tr_trip:>10.5f}  {val_total:>10.5f}  {lr:>9.2e}  {phase}"
        )

        # Track best on MSE-only val loss (more stable than combined)
        monitor = val_mse
        if monitor < best_val:
            best_val, no_improv = monitor, 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"         ✓ saved (val_mse={best_val:.5f})")
        else:
            no_improv += 1
            if no_improv >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val MSE  : {best_val:.5f}")
    print(f"Checkpoint    : {CHECKPOINT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# JSON helper
# ──────────────────────────────────────��──────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Sample generation
# ─────────────────────────────────────────────────────────────────────────────

def build_x_tensor(emotion: str, strength: int) -> torch.Tensor:
    one_hot = torch.zeros(6)
    one_hot[EMOTION_TO_ID[emotion.lower()]] = 1.0
    strength_norm = (strength - STRENGTH_MIN) / (STRENGTH_MAX - STRENGTH_MIN)
    return torch.cat([one_hot, torch.tensor([strength_norm])]).unsqueeze(0)


def generate_samples(model: nn.Module, device: torch.device) -> None:
    model.eval()
    out_dir = Path(SAMPLE_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid: List[Tuple[str, int]] = []
    for emotion in ALLOWED_EMOTIONS:
        for strength in sorted(ALLOWED_STRENGTHS):
            grid.append((emotion, strength))
            if len(grid) >= N_SAMPLE_OUTPUTS:
                break
        if len(grid) >= N_SAMPLE_OUTPUTS:
            break

    print(f"\nGenerating {len(grid)} sample JSON files → {out_dir}/\n")

    with torch.no_grad():
        for counter, (emotion, strength) in enumerate(grid, 1):
            x      = build_x_tensor(emotion, strength).to(device)
            y_pred = model(x).squeeze(0).cpu().numpy()
            jdict  = decode_vector(y_pred)

            fname = out_dir / f"{emotion}_{strength}_{counter}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(jdict, f, indent=2, cls=_NumpyEncoder)

            n_sounds = len(jdict.get("concatenation", {}).get("sound_files", []))
            print(f"  ✓ {fname.name}  ({n_sounds} sound(s))")

    print(f"\nAll samples written to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────��──────

def main() -> None:
    # 1. Load base dataset
    full_dataset = SoundSamplesDataset(dataset_dir=DATASET_DIR, split="train")

    try:
        val_base = SoundSamplesDataset(dataset_dir=DATASET_DIR, split="val")
        train_base = full_dataset
    except FileNotFoundError:
        print("No separate val split — splitting train automatically.")
        n_val   = max(1, int(len(full_dataset) * VALIDATION_SPLIT))
        n_train = len(full_dataset) - n_val
        train_base, val_base = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    # 2. Wrap in triplet datasets
    train_dataset = TripletDataset(full_dataset)   # uses full for triplet mining
    # Val uses plain dataset (no triplets needed for monitoring)
    val_loader = DataLoader(
        val_base,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=triplet_collate if hasattr(val_base, "by_emotion")
                   else _val_collate,
    )

    sampler = full_dataset.get_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=triplet_collate,
    )

    # Val needs triplet format too (y_p / y_n); wrap it
    try:
        val_triplet = TripletDataset(val_base)
    except AttributeError:
        # val_base is a random_split Subset — extract underlying dataset
        val_triplet = TripletDataset(val_base.dataset)  # type: ignore[attr-defined]

    val_loader = DataLoader(
        val_triplet,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=triplet_collate,
    )

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    # 3. Build model
    model = SoundMLP(
        input_dim=7,
        output_dim=VECTOR_SIZE,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: FiLM-SoundMLP  |  params={n_params:,}  |  output={VECTOR_SIZE}")

    # 4. Train
    train_model(model, train_loader, val_loader, DEVICE)

    # 5. Load best & generate
    print(f"\nLoading best checkpoint …")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    generate_samples(model, DEVICE)


# ── tiny helper for plain val batches ────────────────────────────────────────
def _val_collate(batch):
    """Adds dummy y_p / y_n (copies of y) so val loss can reuse run_epoch."""
    x = torch.stack([s["x"] for s in batch])
    y = torch.stack([s["y"] for s in batch])
    return {"x": x, "y": y, "y_p": y, "y_n": y}


if __name__ == "__main__":
    main()