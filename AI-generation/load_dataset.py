"""
Load the encoded dataset for numeric (non-language-model) generative training.

X : float32 tensor, shape (7,)
    [emotion_one_hot (6-dim), strength_norm (1-dim)]

Y : float32 tensor, shape (VECTOR_SIZE,)
    Encoded JSON structure — decode back with decode_vector()

Round-trip example:
    y_vector  = dataset[0]["y"]
    json_dict = decode_vector(y_vector.numpy())
    # json_dict is a valid concatenation JSON ready for audio_synthesis_tool.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import encode/decode helpers from the dataset generator
from dataset_gen import (
    decode_vector,
    encode_x,
    ALLOWED_EMOTIONS,
    EMOTION_TO_ID,
    ALLOWED_STRENGTHS,
    STRENGTH_MIN,
    STRENGTH_MAX,
    VECTOR_SIZE,
)


# ---------------------------
# CONFIG — EDIT THESE VALUES
# ---------------------------

DATASET_DIR  = "dataset_out"
SPLIT: Literal["train", "val"] = "train"
BATCH_SIZE   = 32
NUM_WORKERS  = 0


# ---------------------------
# Dataset
# ---------------------------

class SoundSamplesDataset(Dataset):
    """
    PyTorch Dataset that loads pre-encoded .npy arrays.

    Each item:
        {
            "x": torch.FloatTensor  shape (7,)
                 [emotion one-hot (6) | strength normalized (1)]
            "y": torch.FloatTensor  shape (VECTOR_SIZE,)
                 encoded JSON vector
        }

    To convert a predicted / ground-truth y tensor back to JSON:
        json_dict = decode_vector(sample["y"].numpy())
    """

    def __init__(
        self,
        dataset_dir: str = DATASET_DIR,
        split: Literal["train", "val"] = SPLIT,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        filter_emotion: Optional[List[str]] = None,
        filter_strength: Optional[List[int]] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.transform   = transform

        x_path = self.dataset_dir / f"{split}_x.npy"
        y_path = self.dataset_dir / f"{split}_y.npy"

        for p in (x_path, y_path):
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")

        self.x = torch.from_numpy(np.load(x_path))   # (N, 7)
        self.y = torch.from_numpy(np.load(y_path))   # (N, VECTOR_SIZE)

        assert self.x.shape[0] == self.y.shape[0], "X and Y row counts do not match"
        assert self.y.shape[1] == VECTOR_SIZE, (
            f"Y vector size mismatch: expected {VECTOR_SIZE}, got {self.y.shape[1]}. "
            "Re-generate the dataset if you changed MAX_SOUNDS/MAX_PARTIALS/MAX_FORMANTS."
        )

        # Build index mask for optional filtering
        mask = torch.ones(len(self.x), dtype=torch.bool)

        if filter_emotion:
            keep_ids = {EMOTION_TO_ID[e.lower()] for e in filter_emotion if e.lower() in EMOTION_TO_ID}
            # emotion id is argmax of first 6 dims
            emotion_ids = self.x[:, :6].argmax(dim=1)
            emotion_mask = torch.zeros(len(self.x), dtype=torch.bool)
            for eid in keep_ids:
                emotion_mask |= (emotion_ids == eid)
            mask &= emotion_mask

        if filter_strength:
            # strength_norm is last dim; reverse normalize to get int
            strength_norms = self.x[:, 6]
            strength_mask  = torch.zeros(len(self.x), dtype=torch.bool)
            for s in filter_strength:
                target_norm = (s - STRENGTH_MIN) / (STRENGTH_MAX - STRENGTH_MIN)
                strength_mask |= ((strength_norms - target_norm).abs() < 1e-4)
            mask &= strength_mask

        self.indices = mask.nonzero(as_tuple=True)[0]

        meta_path = self.dataset_dir / "label_maps.json"
        self.meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        print(
            f"SoundSamplesDataset [{split}]: "
            f"{len(self.indices)} samples  "
            f"x_shape={tuple(self.x.shape)}  "
            f"y_shape={tuple(self.y.shape)}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx].item()
        sample = {
            "x": self.x[real_idx],
            "y": self.y[real_idx],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    # ---------------------------
    # Convenience methods
    # ---------------------------

    def decode_y(self, y: torch.Tensor) -> Dict[str, Any]:
        """Convert a Y tensor (model output or ground-truth) back to JSON dict."""
        return decode_vector(y.detach().cpu().numpy())

    def decode_x(self, x: torch.Tensor) -> Dict[str, Any]:
        """Convert an X tensor back to human-readable emotion + strength."""
        emotion_id = int(x[:6].argmax().item())
        strength_norm = float(x[6].item())
        strength = round(strength_norm * (STRENGTH_MAX - STRENGTH_MIN) + STRENGTH_MIN)
        return {
            "emotion":      ALLOWED_EMOTIONS[emotion_id],
            "emotion_id":   emotion_id,
            "strength":     strength,
            "strength_norm": strength_norm,
        }

    def emotion_distribution(self) -> Dict[str, int]:
        emotion_ids = self.x[self.indices, :6].argmax(dim=1).tolist()
        counts = {e: 0 for e in ALLOWED_EMOTIONS}
        for eid in emotion_ids:
            counts[ALLOWED_EMOTIONS[eid]] += 1
        return counts

    def strength_distribution(self) -> Dict[int, int]:
        norms  = self.x[self.indices, 6].tolist()
        counts = {s: 0 for s in sorted(ALLOWED_STRENGTHS)}
        for n in norms:
            s = round(n * (STRENGTH_MAX - STRENGTH_MIN) + STRENGTH_MIN)
            if s in counts:
                counts[s] += 1
        return counts


# ---------------------------
# Collate (default is fine, but provided for clarity)
# ---------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.stack([s["x"] for s in batch]),   # (B, 7)
        "y": torch.stack([s["y"] for s in batch]),   # (B, VECTOR_SIZE)
    }


# ---------------------------
# Quick smoke-test / round-trip check
# ---------------------------

def main() -> None:
    dataset = SoundSamplesDataset(
        dataset_dir=DATASET_DIR,
        split=SPLIT,
        # filter_emotion=["anger", "sad"],
        # filter_strength=[4, 5],
    )

    print("\nEmotion distribution:", dataset.emotion_distribution())
    print("Strength distribution:", dataset.strength_distribution())
    print(f"Vector size Y: {VECTOR_SIZE}")

    sample = dataset[0]
    print("\nSample[0] x shape:", sample["x"].shape)
    print("Sample[0] y shape:", sample["y"].shape)
    print("Sample[0] decoded x:", dataset.decode_x(sample["x"]))

    # Round-trip: tensor -> JSON -> re-check structure
    json_dict = dataset.decode_y(sample["y"])
    n_sounds  = len(json_dict["concatenation"]["sound_files"])
    print(f"Sample[0] decoded JSON: {n_sounds} sound(s)")
    print(json.dumps(json_dict["concatenation"]["sound_files"][0]["global"], indent=2))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    print("\nBatch x shape:", batch["x"].shape)
    print("Batch y shape:", batch["y"].shape)
    print("First batch item x decoded:", dataset.decode_x(batch["x"][0]))
    print("First batch item x:", batch["x"][0].numpy())

    # Decode first sample in batch back to JSON
    #first_json = decode_vector(batch["y"][0].numpy())
    #print("First batch item y decoded:", first_json)
    #print("First batch item y:", batch["y"][0].numpy())


if __name__ == "__main__":
    main()