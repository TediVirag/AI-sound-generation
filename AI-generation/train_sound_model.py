"""
Train a FiLM-conditioned two-headed MLP on SoundSamplesDataset.

Key features:
  - FiLM conditioning: emotion modulates every hidden layer directly
  - Two-headed: output head (VECTOR_SIZE) + embed head (EMBED_DIM)
  - Triplet loss on 32-dim L2-normalised embeddings (stable, doesn't interfere with MSE)
  - MaskedMSELoss: ignores zero-padded positions in the Y vector
  - LR warmup + cosine annealing

Input  X : float32 (7,)   — [emotion_one_hot (6) | strength_norm (1)]
Output Y : float32 (VECTOR_SIZE,) — encoded JSON vector

Usage:
    python train_sound_film_mlp2.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from sound_samples_dataset import SoundSamplesDataset, DATASET_DIR
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
EPOCHS           = 150 # 0 epochs means training won't run only sample generation
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-5
VALIDATION_SPLIT = 0.15
PATIENCE         = 25

HIDDEN_DIMS  = [512, 1024, 1024, 512]
DROPOUT      = 0.15
EMBED_DIM    = 32

MSE_WEIGHT     = 1.0
TRIPLET_WEIGHT = 0.15
TRIPLET_MARGIN = 1.0

CHECKPOINT_PATH   = "best_sound_model.pt"
SAMPLE_OUTPUT_DIR = "generated_samples"
N_SAMPLE_OUTPUTS  = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Masked MSE Loss
# ─────────────────────────────────────────────────────────────────────────────

class MaskedMSELoss(nn.Module):
    """
    MSE loss that only computes error on positions that are non-zero
    in at least one training sample. Zero-padded positions (unused
    partial/formant/sound slots) are excluded from the gradient entirely.
    """
    def __init__(self, active_mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", active_mask.float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = (pred - target) ** 2
        masked = sq_err * self.mask
        return masked.sum() / (self.mask.sum() * pred.shape[0])


def build_active_mask(
    train_loader: DataLoader,
    vector_size:  int,
    device:       torch.device,
) -> torch.Tensor:
    ever_nonzero = torch.zeros(vector_size, dtype=torch.bool)
    for batch in train_loader:
        y = batch["y"]
        ever_nonzero |= (y != 0.0).any(dim=0)
        if ever_nonzero.all():
            break
    n_active  = ever_nonzero.sum().item()
    n_padding = vector_size - n_active
    print(f"  Active positions  : {n_active} / {vector_size}")
    print(f"  Padding positions : {n_padding} (excluded from loss)")
    return ever_nonzero.to(device)

# ─────────────────────────────────────────────────────────────────────────────────
# FiLM layer
# ─────────────────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Learns scale (γ) and shift (β) from the condition vector c,
    then applies: out = γ(c) * x + β(c)
    """
    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta  = nn.Linear(condition_dim, feature_dim)
        nn.init.ones_(self.gamma.weight);  nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.gamma(c) * x + self.beta(c)

# ─────────────────────────────────────────────────────────────────────────────
# Two-headed FiLM MLP
# ─────────────────────────────────────────────────────────────────────────────

class SoundMLP(nn.Module):
    """
    FiLM-conditioned MLP with two heads:
      • output_head : Linear → VECTOR_SIZE   (reconstruction, MaskedMSE loss)
      • embed_head  : Linear → EMBED_DIM     (emotion separation, triplet loss)

    Architecture per block:
        Linear → LayerNorm → GELU → FiLM(emotion) → Dropout
    """

    def __init__(
        self,
        input_dim:     int = 7,
        output_dim:    int = VECTOR_SIZE,
        hidden_dims:   List[int] = None,
        dropout:       float = DROPOUT,
        condition_dim: int = 6,
        embed_dim:     int = EMBED_DIM,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        self.condition_dim = condition_dim
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.linears = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.films   = nn.ModuleList()
        self.drops   = nn.ModuleList()

        prev = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(prev, h))
            self.norms.append(nn.LayerNorm(h))
            self.films.append(FiLMLayer(h, 64))
            self.drops.append(nn.Dropout(dropout))
            prev = h

        self.output_head = nn.Linear(prev, output_dim)
        self.embed_head  = nn.Linear(prev, embed_dim)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        emotion = x[:, :self.condition_dim]
        c = self.condition_proj(emotion)
        h = x
        for linear, norm, film, drop in zip(
            self.linears, self.norms, self.films, self.drops
        ):
            h = linear(h)
            h = norm(h)
            h = F.gelu(h)
            h = film(h, c)
            h = drop(h)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h   = self.trunk(x)
        out = self.output_head(h)
        emb = F.normalize(self.embed_head(h), dim=-1)   # L2-normalised embeddings
        return out, emb

# ─────────────────────────────────────────────────────────────────────────────
# Triplet dataset — triplets on x vectors for embed head
# ─────────────────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Yields (anchor_x, anchor_y, positive_x, negative_x) triplets.
    Triplet loss is computed on embed_head(x) — NOT on raw y vectors.
    """

    def __init__(self, base: SoundSamplesDataset) -> None:
        self.base = base
        self.by_emotion: Dict[int, List[int]] = {}
        emotion_ids = base.x[base.indices, :6].argmax(dim=1).tolist()
        for local_i, eid in enumerate(emotion_ids):
            self.by_emotion.setdefault(eid, []).append(local_i)
        self.all_emotions = list(self.by_emotion.keys())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = self.base[idx]
        eid    = int(anchor["x"][:6].argmax().item())

        # Positive: strictly different index, same emotion
        pos_pool = [i for i in self.by_emotion[eid] if i != idx]
        pos_idx  = random.choice(pos_pool) if pos_pool else idx
        pos      = self.base[pos_idx]

        # Negative: different emotion entirely
        neg_eid = random.choice([e for e in self.all_emotions if e != eid])
        neg     = self.base[random.choice(self.by_emotion[neg_eid])]

        return {
            "x":   anchor["x"],
            "y":   anchor["y"],
            "x_p": pos["x"],     # positive x — for embed triplet
            "x_n": neg["x"],     # negative x — for embed triplet
        }


def triplet_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}

# ─────────────────────────────────────────────────────────────────────────────────
# Train / val epoch
# ─────────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    mse_crit:  nn.Module,
    trip_crit: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    train:     bool,
) -> Tuple[float, float, float]:
    model.train(train)
    total_mse, total_trip, total = 0.0, 0.0, 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x   = batch["x"].to(device)
            y   = batch["y"].to(device)
            x_p = batch["x_p"].to(device)
            x_n = batch["x_n"].to(device)

            pred,  emb_a = model(x)
            _,     emb_p = model(x_p)
            _,     emb_n = model(x_n)

            mse  = mse_crit(pred, y)
            trip = trip_crit(emb_a, emb_p, emb_n)
            loss = MSE_WEIGHT * mse + TRIPLET_WEIGHT * trip

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

            total_mse  += mse.item()
            total_trip += trip.item()
            total      += loss.item()
            n          += 1

    n = max(n, 1)
    return total / n, total_mse / n, total_trip / n

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
) -> None:
    print("\nBuilding active position mask...")
    active_mask = build_active_mask(train_loader, VECTOR_SIZE, device)

    mse_crit  = MaskedMSELoss(active_mask)
    trip_crit = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS - 10, 1), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[10]
    )

    best_val, no_improv = float("inf"), 0

    print(f"\nTraining on {device}  |  epochs={EPOCHS}  |  patience={PATIENCE}")
    print(f"Loss = {MSE_WEIGHT}×MaskedMSE + {TRIPLET_WEIGHT}×Triplet(margin={TRIPLET_MARGIN})")
    print(f"Triplet on {EMBED_DIM}-dim L2-normalised embeddings\n")
    print(f"{'Epoch':>5}  {'Total':>10}  {'MSE':>10}  {'Triplet':>10}  {'Val MSE':>10}  {'LR':>9}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        tr_total, tr_mse, tr_trip = run_epoch(
            model, train_loader, mse_crit, trip_crit, optimizer, device, train=True
        )
        val_total, val_mse, val_trip = run_epoch(
            model, val_loader, mse_crit, trip_crit, None, device, train=False
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>5}  {tr_total:>10.5f}  {tr_mse:>10.5f}  "
            f"{tr_trip:>10.5f}  {val_mse:>10.5f}  {lr:>9.2e}"
        )

        if val_mse < best_val:
            best_val, no_improv = val_mse, 0
            import shutil
            tmp = CHECKPOINT_PATH + ".tmp"
            torch.save(model.state_dict(), tmp)
            shutil.move(tmp, CHECKPOINT_PATH)
            print(f"         ✓ saved (val_mse={best_val:.5f})")
        else:
            no_improv += 1
            if no_improv >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val MSE : {best_val:.5f}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# JSON helper
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_x_tensor(emotion: str, strength: int) -> torch.Tensor:
    one_hot = torch.zeros(6)
    one_hot[EMOTION_TO_ID[emotion.lower()]] = 1.0
    norm = (strength - STRENGTH_MIN) / (STRENGTH_MAX - STRENGTH_MIN)
    return torch.cat([one_hot, torch.tensor([norm])]).unsqueeze(0)


def generate_samples(model: nn.Module, device: torch.device) -> None:
    # Use model.train() instead of model.eval() if you want Dropout 
    # to stay active and produce variations for the exact same emotion/strength!
    model.eval() 
    
    out_dir = Path(SAMPLE_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create the base 30 combinations (6 emotions * 5 strengths)
    base_combos = [(e, s) for e in ALLOWED_EMOTIONS for s in sorted(ALLOWED_STRENGTHS)]
    
    # 2. Cycle through them until we reach N_SAMPLE_OUTPUTS (e.g., 150)
    grid = list(itertools.islice(itertools.cycle(base_combos), N_SAMPLE_OUTPUTS))

    print(f"\nGenerating {len(grid)} sample JSON files → {out_dir}/\n")

    with torch.no_grad():
        for counter, (emotion, strength) in enumerate(grid, 1):
            x = build_x_tensor(emotion, strength).to(device)
            
            # Optional: Add tiny noise to the 'strength' dimension so repeated sounds are slightly unique
            x[0, 6] += torch.randn(1).item() * 0.05 
            
            out, _ = model(x)
            y_pred  = out.squeeze(0).cpu().numpy()
            jdict   = decode_vector(y_pred)

            fname = out_dir / f"{emotion}_{strength}_{counter}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(jdict, f, indent=2, cls=_NumpyEncoder)

            n_sounds = len(jdict.get("concatenation", {}).get("sound_files", []))
            print(f"  ✓ {fname.name}  ({n_sounds} sound(s))")

    print(f"\nAll samples written to '{out_dir}/'")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load dataset
    full_dataset = SoundSamplesDataset(dataset_dir=DATASET_DIR, split="train")

    try:
        val_base = SoundSamplesDataset(dataset_dir=DATASET_DIR, split="val")
    except FileNotFoundError:
        print("No val split found — splitting train automatically.")
        n_val   = max(1, int(len(full_dataset) * VALIDATION_SPLIT))
        n_train = len(full_dataset) - n_val
        full_dataset, val_base = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        full_dataset = full_dataset.dataset

    # 2. Wrap in triplet datasets
    train_triplet = TripletDataset(full_dataset)
    val_triplet   = TripletDataset(
        val_base if isinstance(val_base, SoundSamplesDataset)
        else val_base.dataset
    )

    sampler = full_dataset.get_sampler()

    train_loader = DataLoader(
        train_triplet,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=triplet_collate,
    )
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
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: FiLM-SoundMLP (2-head)  |  params={n_params:,}  |  output={VECTOR_SIZE}  |  embed={EMBED_DIM}")

    # 4. Train
    train_model(model, train_loader, val_loader, DEVICE)

    # 5. Load best checkpoint and generate samples
    print(f"\nLoading best checkpoint from '{CHECKPOINT_PATH}' ...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    generate_samples(model, DEVICE)


if __name__ == "__main__":
    main()