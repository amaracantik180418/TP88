#!/usr/bin/env python3
"""
TP88 - AI Training Software Bot (TensorProxima companion app).
Single-file app: run registry, epochs, checkpoints, training loop, metrics, CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

TP88_VERSION = (8, 8)
TP88_RUN_PREFIX = "tp88_"
TP88_LOSS_SCALE = 1_000_000_000
TP88_MAX_EPOCHS = 50_000
TP88_MAX_CHECKPOINTS = 2_000
TP88_GRADIENT_CLIP_NORM = 5.0
TP88_DEFAULT_LR = 1e-3
TP88_DEFAULT_BATCH = 32
TP88_DEFAULT_EPOCHS = 100
TP88_CHECKPOINT_EVERY = 5
TP88_SEED_BASE = 0x8F4B2C1E

# -----------------------------------------------------------------------------
# EXCEPTIONS
# -----------------------------------------------------------------------------


class TP88RunNotFoundError(Exception):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run not found: {run_id}")
        self.run_id = run_id


class TP88EpochIndexError(Exception):
    def __init__(self, index: int, maximum: int) -> None:
        super().__init__(f"Epoch index {index} out of range [0, {maximum})")
        self.index = index
        self.maximum = maximum


class TP88CheckpointError(Exception):
    pass


class TP88ConfigValidationError(Exception):
    def __init__(self, field_name: str) -> None:
        super().__init__(f"Invalid config: {field_name}")
        self.field_name = field_name


class TP88GradientExplosionError(Exception):
    def __init__(self, norm: float) -> None:
        super().__init__(f"Gradient norm too large: {norm}")
        self.norm = norm


class TP88DatasetEmptyError(Exception):
    pass


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------


@dataclass
class TrainingRunRecord:
    run_id: str
    submitter_id: str
    epoch_count: int
    config_hash: bytes
    registered_at: float
    archived: bool = False
    epochs_recorded: int = 0
    checkpoints_anchored: int = 0


@dataclass
class EpochRecord:
    run_id: str
    epoch_index: int
    loss_scaled: int
    gradient_root: bytes
    recorded_at: float

    @property
    def loss(self) -> float:
        return self.loss_scaled / TP88_LOSS_SCALE


@dataclass
class CheckpointRecord:
    run_id: str
    checkpoint_index: int
    state_hash: bytes
    anchored_at: float


@dataclass
class TrainingConfig:
    max_epochs: int = TP88_DEFAULT_EPOCHS
    batch_size: int = TP88_DEFAULT_BATCH
    learning_rate: float = TP88_DEFAULT_LR
    gradient_clip_norm: float = TP88_GRADIENT_CLIP_NORM
    checkpoint_every_epochs: int = TP88_CHECKPOINT_EVERY
    random_seed: int = 0
    optimizer_name: str = "Adam"
    loss_name: str = "MSE"

    def __post_init__(self) -> None:
        if self.random_seed == 0:
            self.random_seed = TP88_SEED_BASE + int(time.time() * 1e6) % (2**32)


@dataclass
class EpochMetrics:
    epoch_index: int
    loss: float
    duration_ms: float
    batches_processed: int

    def __str__(self) -> str:
        return (
            f"EpochMetrics(epoch={self.epoch_index}, loss={self.loss:.6f}, "
            f"duration_ms={self.duration_ms}, batches={self.batches_processed})"
        )


# -----------------------------------------------------------------------------
# RUN REGISTRY
# -----------------------------------------------------------------------------


class RunRegistry:
    def __init__(self) -> None:
        self._runs: Dict[str, TrainingRunRecord] = {}
        self._epochs: Dict[str, List[EpochRecord]] = {}
        self._checkpoints: Dict[str, List[CheckpointRecord]] = {}
        self._run_id_order: List[str] = []

    def register_run(
        self,
        submitter_id: str,
        epoch_count: int,
        config_hash: Optional[bytes] = None,
    ) -> str:
        run_id = f"{TP88_RUN_PREFIX}{uuid.uuid4().hex[:16]}"
        if epoch_count <= 0 or epoch_count > TP88_MAX_EPOCHS:
            raise TP88ConfigValidationError("epoch_count")
        h = config_hash or bytes(32)
        rec = TrainingRunRecord(
            run_id=run_id,
            submitter_id=submitter_id,
            epoch_count=epoch_count,
            config_hash=h,
            registered_at=time.time(),
        )
        self._runs[run_id] = rec
        self._run_id_order.append(run_id)
        self._epochs[run_id] = []
        self._checkpoints[run_id] = []
        return run_id

    def get_run(self, run_id: str) -> TrainingRunRecord:
        if run_id not in self._runs:
            raise TP88RunNotFoundError(run_id)
        return self._runs[run_id]

    def record_epoch(
        self,
        run_id: str,
        epoch_index: int,
        loss_scaled: int,
        gradient_root: bytes,
    ) -> None:
        r = self.get_run(run_id)
        if r.archived:
