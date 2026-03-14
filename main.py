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
            raise TP88CheckpointError("Run archived")
        if epoch_index >= r.epoch_count:
            raise TP88EpochIndexError(epoch_index, r.epoch_count)
        if r.epochs_recorded != epoch_index:
            raise TP88CheckpointError("Epoch order")
        rec = EpochRecord(
            run_id=run_id,
            epoch_index=epoch_index,
            loss_scaled=loss_scaled,
            gradient_root=gradient_root or bytes(32),
            recorded_at=time.time(),
        )
        self._epochs[run_id].append(rec)
        r.epochs_recorded += 1

    def anchor_checkpoint(
        self,
        run_id: str,
        checkpoint_index: int,
        state_hash: bytes,
    ) -> None:
        r = self.get_run(run_id)
        if r.archived:
            raise TP88CheckpointError("Run archived")
        if checkpoint_index >= TP88_MAX_CHECKPOINTS:
            raise TP88CheckpointError("Checkpoint index out of range")
        rec = CheckpointRecord(
            run_id=run_id,
            checkpoint_index=checkpoint_index,
            state_hash=state_hash or bytes(32),
            anchored_at=time.time(),
        )
        self._checkpoints[run_id].append(rec)
        r.checkpoints_anchored += 1

    def archive_run(self, run_id: str) -> None:
        self.get_run(run_id).archived = True

    def get_epochs(self, run_id: str) -> List[EpochRecord]:
        return list(self._epochs.get(run_id, []))

    def get_checkpoints(self, run_id: str) -> List[CheckpointRecord]:
        return list(self._checkpoints.get(run_id, []))

    def get_all_run_ids(self) -> List[str]:
        return list(self._run_id_order)

    def total_runs(self) -> int:
        return len(self._runs)


# -----------------------------------------------------------------------------
# LOSS FUNCTIONS
# -----------------------------------------------------------------------------


class LossFunction:
    def compute(self, predicted: Sequence[float], target: Sequence[float]) -> float:
        raise NotImplementedError

    def gradient(
        self,
        predicted: Sequence[float],
        target: Sequence[float],
        gradient_out: List[float],
    ) -> None:
        raise NotImplementedError

    def name(self) -> str:
        return "Loss"


class MSELoss(LossFunction):
    def compute(self, predicted: Sequence[float], target: Sequence[float]) -> float:
        n = len(predicted)
        return sum((p - t) ** 2 for p, t in zip(predicted, target)) / n

    def gradient(
        self,
        predicted: Sequence[float],
        target: Sequence[float],
        gradient_out: List[float],
    ) -> None:
        n = len(predicted)
        for i, (p, t) in enumerate(zip(predicted, target)):
            gradient_out[i] = 2.0 * (p - t) / n

    def name(self) -> str:
        return "MSE"


class CrossEntropyLoss(LossFunction):
    def compute(self, predicted: Sequence[float], target: Sequence[float]) -> float:
        n = len(predicted)
        return -sum(
            t * math.log(max(1e-15, min(1 - 1e-15, p)))
            for p, t in zip(predicted, target)
        ) / n

    def gradient(
        self,
        predicted: Sequence[float],
        target: Sequence[float],
        gradient_out: List[float],
    ) -> None:
        n = len(predicted)
        for i, (p, t) in enumerate(zip(predicted, target)):
            p = max(1e-15, min(1 - 1e-15, p))
            gradient_out[i] = -(t / p) / n

    def name(self) -> str:
        return "CrossEntropy"


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def compute(self, predicted: Sequence[float], target: Sequence[float]) -> float:
        n = len(predicted)
        total = 0.0
        for p, t in zip(predicted, target):
            d = p - t
            ad = abs(d)
            total += (
                0.5 * d * d if ad <= self.delta else self.delta * (ad - 0.5 * self.delta)
            )
        return total / n

    def gradient(
        self,
        predicted: Sequence[float],
        target: Sequence[float],
        gradient_out: List[float],
    ) -> None:
        n = len(predicted)
        for i, (p, t) in enumerate(zip(predicted, target)):
            d = p - t
            if abs(d) <= self.delta:
                gradient_out[i] = d / n
            else:
                gradient_out[i] = (self.delta * (1 if d > 0 else -1)) / n

    def name(self) -> str:
        return "Huber"


# -----------------------------------------------------------------------------
# OPTIMIZERS
# -----------------------------------------------------------------------------


class Optimizer:
    def step(
        self,
        params: List[float],
        gradients: List[float],
        step_index: int,
    ) -> None:
        raise NotImplementedError

    def name(self) -> str:
        return "Optimizer"


class SGDOptimizer(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.9, param_len: int = 0) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity = [0.0] * param_len

    def step(
        self,
        params: List[float],
        gradients: List[float],
        step_index: int,
    ) -> None:
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] + gradients[i]
            params[i] -= self.lr * self.velocity[i]

    def name(self) -> str:
        return "SGD"


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        param_len: int = 0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * param_len
        self.v = [0.0] * param_len
        self.t = 0

    def step(
        self,
        params: List[float],
        gradients: List[float],
        step_index: int,
    ) -> None:
        self.t += 1
        for i in range(len(params)):
            g = gradients[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def name(self) -> str:
        return "Adam"


class RMSpropOptimizer(Optimizer):
    def __init__(self, lr: float, decay: float = 0.99, param_len: int = 0) -> None:
        self.lr = lr
        self.decay = decay
        self.cache = [0.0] * param_len

    def step(
        self,
        params: List[float],
        gradients: List[float],
        step_index: int,
    ) -> None:
        for i in range(len(params)):
            g = gradients[i]
            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * g * g
            params[i] -= self.lr * g / (math.sqrt(self.cache[i]) + 1e-8)

    def name(self) -> str:
        return "RMSprop"


# -----------------------------------------------------------------------------
# GRADIENT UTILS
# -----------------------------------------------------------------------------


def gradient_norm(g: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in g))


def gradient_clip(g: List[float], max_norm: float) -> None:
    n = gradient_norm(g)
    if n > max_norm and n > 0:
        scale = max_norm / n
        for i in range(len(g)):
            g[i] *= scale


def hash_for_root(gradient: Sequence[float]) -> bytes:
    buf = struct.pack(f"{len(gradient)}d", *gradient)
    return hashlib.sha256(buf).digest()


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------


class Dataset:
    def size(self) -> int:
        raise NotImplementedError

    def feature_dim(self) -> int:
        raise NotImplementedError

    def target_dim(self) -> int:
        raise NotImplementedError

    def get_batch(
        self,
        start_idx: int,
        count: int,
        features_out: List[List[float]],
        targets_out: List[List[float]],
    ) -> None:
        raise NotImplementedError


class ArrayDataset(Dataset):
    def __init__(
        self,
        features: List[List[float]],
        targets: List[List[float]],
        seed: int = 0,
    ) -> None:
        if len(features) != len(targets) or len(features) == 0:
            raise TP88DatasetEmptyError()
        self.features = features
        self.targets = targets
        self.rng = random.Random(seed)

    def size(self) -> int:
        return len(self.features)

    def feature_dim(self) -> int:
        return len(self.features[0])

    def target_dim(self) -> int:
        return len(self.targets[0])

    def get_batch(
        self,
        start_idx: int,
        count: int,
        features_out: List[List[float]],
        targets_out: List[List[float]],
    ) -> None:
        n = min(count, len(self.features) - start_idx)
        for i in range(n):
            features_out[i][:] = self.features[start_idx + i]
            targets_out[i][:] = self.targets[start_idx + i]

    def shuffled_indices(self) -> List[int]:
        idx = list(range(len(self.features)))
        self.rng.shuffle(idx)
        return idx


# -----------------------------------------------------------------------------
# MODEL (simple linear)
# -----------------------------------------------------------------------------


class Model:
    def forward(self, input_batch: List[List[float]], output_batch: List[List[float]]) -> None:
        raise NotImplementedError

    def backward(
        self,
        input_batch: List[List[float]],
        output_grad: List[List[float]],
        param_grad: List[List[float]],
    ) -> None:
        raise NotImplementedError

    def get_params(self) -> List[float]:
        raise NotImplementedError

    def set_params(self, params: Sequence[float]) -> None:
        raise NotImplementedError

    def param_count(self) -> int:
        raise NotImplementedError


class LinearModel(Model):
    def __init__(self, in_dim: int, out_dim: int, rng: random.Random) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        param_len = out_dim * (in_dim + 1)
        scale = 1.0 / math.sqrt(in_dim + 1)
        self.params = [(rng.random() * 2 - 1) * scale for _ in range(param_len)]

    def param_count(self) -> int:
        return len(self.params)

    def get_params(self) -> List[float]:
        return list(self.params)

    def set_params(self, params: Sequence[float]) -> None:
        self.params[:] = params[: len(self.params)]

    def forward(self, input_batch: List[List[float]], output_batch: List[List[float]]) -> None:
        batch = len(input_batch)
        for b in range(batch):
            for o in range(self.out_dim):
                base = o * (self.in_dim + 1)
                s = self.params[base + self.in_dim]
                for i in range(self.in_dim):
                    s += input_batch[b][i] * self.params[base + i]
                output_batch[b][o] = s

    def backward(
        self,
        input_batch: List[List[float]],
        output_grad: List[List[float]],
        param_grad: List[List[float]],
    ) -> None:
        batch = len(input_batch)
        rows = self.out_dim
        cols = self.in_dim + 1
        for r in range(rows):
            for c in range(cols):
                param_grad[r][c] = 0.0
        for b in range(batch):
            for o in range(self.out_dim):
                g = output_grad[b][o]
                for i in range(self.in_dim):
                    param_grad[o][i] += g * input_batch[b][i]
                param_grad[o][self.in_dim] += g
        for o in range(self.out_dim):
            for i in range(self.in_dim + 1):
                param_grad[o][i] /= batch


# -----------------------------------------------------------------------------
# LOSS & OPTIMIZER FACTORIES
# -----------------------------------------------------------------------------


def create_loss(name: str, **kwargs: Any) -> LossFunction:
    if name == "MSE":
        return MSELoss()
    if name == "CrossEntropy":
        return CrossEntropyLoss()
    if name == "Huber":
        return HuberLoss(delta=kwargs.get("delta", 1.0))
    return MSELoss()


def create_optimizer(
    name: str,
    lr: float,
    param_len: int,
    **kwargs: Any,
) -> Optimizer:
    if name == "SGD":
        return SGDOptimizer(lr, momentum=kwargs.get("momentum", 0.9), param_len=param_len)
    if name == "Adam":
        return AdamOptimizer(
            lr,
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.999),
            eps=kwargs.get("eps", 1e-8),
            param_len=param_len,
        )
    if name == "RMSprop":
        return RMSpropOptimizer(lr, decay=kwargs.get("decay", 0.99), param_len=param_len)
    return AdamOptimizer(lr, param_len=param_len)


# -----------------------------------------------------------------------------
# TRAINER BOT
# -----------------------------------------------------------------------------


class TrainerBot:
    def __init__(
        self,
        registry: RunRegistry,
        config: TrainingConfig,
        loss_fn: LossFunction,
        optimizer: Optimizer,
        model: Model,
        dataset: Dataset,
    ) -> None:
        self.registry = registry
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.dataset = dataset
        self.gradient_clip_norm = config.gradient_clip_norm

    def _hash_config(self, c: TrainingConfig) -> bytes:
        s = f"{c.max_epochs}|{c.batch_size}|{c.learning_rate}|{c.optimizer_name}|{c.loss_name}"
        return hashlib.sha256(s.encode()).digest()

    def start_run(self, submitter_id: str) -> str:
        config_hash = self._hash_config(self.config)
        return self.registry.register_run(
            submitter_id,
            self.config.max_epochs,
            config_hash,
        )

    def run_training(self, run_id: str) -> None:
        r = self.registry.get_run(run_id)
        batch_size = self.config.batch_size
        feature_dim = self.dataset.feature_dim()
        target_dim = self.dataset.target_dim()
        n_samples = self.dataset.size()
        batch_count = (n_samples + batch_size - 1) // batch_size
        batch_features = [[0.0] * feature_dim for _ in range(batch_size)]
        batch_targets = [[0.0] * target_dim for _ in range(batch_size)]
        batch_output = [[0.0] * target_dim for _ in range(batch_size)]
        output_grad = [[0.0] * target_dim for _ in range(batch_size)]
        param_grad = [
            [0.0] * (feature_dim + 1) for _ in range(target_dim)
        ]
        global_step = 0
        for epoch in range(self.config.max_epochs):
            t0 = time.perf_counter()
            epoch_loss = 0.0
            indices = (
                self.dataset.shuffled_indices()
                if isinstance(self.dataset, ArrayDataset)
                else list(range(n_samples))
            )
            for b in range(batch_count):
                start = b * batch_size
                length = min(batch_size, n_samples - start)
                if length <= 0:
                    continue
                for i in range(length):
                    idx = indices[start + i]
                    self.dataset.get_batch(
                        idx, 1,
                        [batch_features[i]],
                        [batch_targets[i]],
                    )
                self.model.forward(batch_features, batch_output)
                batch_loss = 0.0
                for i in range(length):
                    batch_loss += self.loss_fn.compute(batch_output[i], batch_targets[i])
                    grad_out = [0.0] * target_dim
                    self.loss_fn.gradient(batch_output[i], batch_targets[i], grad_out)
                    output_grad[i][:] = grad_out
                batch_loss /= length
                epoch_loss += batch_loss
                params = self.model.get_params()
                self.model.backward(batch_features, output_grad, param_grad)
                flat_grad = []
                for row in param_grad:
                    flat_grad.extend(row)
                gradient_clip(flat_grad, self.gradient_clip_norm)
                if gradient_norm(flat_grad) > self.gradient_clip_norm * 100:
                    raise TP88GradientExplosionError(gradient_norm(flat_grad))
                self.optimizer.step(params, flat_grad, global_step)
                self.model.set_params(params)
                global_step += 1
            epoch_loss /= batch_count
            loss_scaled = int(epoch_loss * TP88_LOSS_SCALE)
            gradient_root = hash_for_root(self.model.get_params())
            self.registry.record_epoch(run_id, epoch, loss_scaled, gradient_root)
            if (epoch + 1) % self.config.checkpoint_every_epochs == 0:
                buf = struct.pack(f"{self.model.param_count()}d", *self.model.get_params())
                state_hash = hashlib.sha256(buf).digest()
                self.registry.anchor_checkpoint(
                    run_id,
                    (epoch + 1) // self.config.checkpoint_every_epochs - 1,
                    state_hash,
                )
            duration_ms = (time.perf_counter() - t0) * 1000
            metrics = EpochMetrics(epoch, epoch_loss, duration_ms, batch_count)
            if epoch % 10 == 0:
                print(metrics)


# -----------------------------------------------------------------------------
# SYNTHETIC DATA
# -----------------------------------------------------------------------------


def generate_synthetic_linear(
    num_samples: int,
    feature_dim: int,
    target_dim: int,
    seed: int,
) -> ArrayDataset:
    rng = random.Random(seed)
    w = [[rng.random() * 2 - 1 for _ in range(feature_dim + 1)] for _ in range(target_dim)]
    features = [[rng.random() * 2 - 1 for _ in range(feature_dim)] for _ in range(num_samples)]
    targets = []
    for i in range(num_samples):
        row = []
        for o in range(target_dim):
            s = w[o][feature_dim]
            for j in range(feature_dim):
                s += features[i][j] * w[o][j]
            row.append(s + 0.1 * rng.gauss(0, 1))
        targets.append(row)
    return ArrayDataset(features, targets, rng.randint(0, 2**31))


def generate_synthetic_random(
    num_samples: int,
    feature_dim: int,
    target_dim: int,
    seed: int,
) -> ArrayDataset:
    rng = random.Random(seed)
    features = [[rng.random() for _ in range(feature_dim)] for _ in range(num_samples)]
    targets = [[rng.random() for _ in range(target_dim)] for _ in range(num_samples)]
    return ArrayDataset(features, targets, rng.randint(0, 2**31))


# -----------------------------------------------------------------------------
# CONFIG SERIALIZATION
# -----------------------------------------------------------------------------


def config_to_json(c: TrainingConfig) -> str:
    return json.dumps({
        "max_epochs": c.max_epochs,
        "batch_size": c.batch_size,
        "learning_rate": c.learning_rate,
        "gradient_clip_norm": c.gradient_clip_norm,
        "checkpoint_every_epochs": c.checkpoint_every_epochs,
        "random_seed": c.random_seed,
        "optimizer_name": c.optimizer_name,
        "loss_name": c.loss_name,
    })


def config_from_json(s: str) -> TrainingConfig:
    d = json.loads(s)
    return TrainingConfig(
        max_epochs=d.get("max_epochs", TP88_DEFAULT_EPOCHS),
        batch_size=d.get("batch_size", TP88_DEFAULT_BATCH),
        learning_rate=d.get("learning_rate", TP88_DEFAULT_LR),
        gradient_clip_norm=d.get("gradient_clip_norm", TP88_GRADIENT_CLIP_NORM),
        checkpoint_every_epochs=d.get("checkpoint_every_epochs", TP88_CHECKPOINT_EVERY),
        random_seed=d.get("random_seed", TP88_SEED_BASE),
        optimizer_name=d.get("optimizer_name", "Adam"),
        loss_name=d.get("loss_name", "MSE"),
    )


# -----------------------------------------------------------------------------
# RUN SUMMARY & EXPORT
# -----------------------------------------------------------------------------


@dataclass
class RunSummary:
    run_id: str
    total_epochs: int
    final_loss: float
    checkpoints_anchored: int
    duration_ms: float

    @staticmethod
    def from_registry(registry: RunRegistry, run_id: str) -> "RunSummary":
        r = registry.get_run(run_id)
        epochs = registry.get_epochs(run_id)
        checkpoints = registry.get_checkpoints(run_id)
        final_loss = epochs[-1].loss if epochs else float("nan")
        start = r.registered_at
        end = epochs[-1].recorded_at if epochs else start
        return RunSummary(
            run_id=run_id,
            total_epochs=len(epochs),
            final_loss=final_loss,
            checkpoints_anchored=len(checkpoints),
            duration_ms=(end - start) * 1000,
        )


def export_epochs_csv(registry: RunRegistry, run_id: str, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch_index,loss_scaled,loss,recorded_at"]
    for e in registry.get_epochs(run_id):
        lines.append(f"{e.epoch_index},{e.loss_scaled},{e.loss:.10f},{e.recorded_at}")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_checkpoints_csv(registry: RunRegistry, run_id: str, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["checkpoint_index,anchored_at"]
    for c in registry.get_checkpoints(run_id):
        lines.append(f"{c.checkpoint_index},{c.anchored_at}")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# EARLY STOPPING & VALIDATION
# -----------------------------------------------------------------------------


class EarlyStoppingHandler:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.wait_count = 0
        self.best_loss = float("inf")

    def should_stop(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
            return False
        self.wait_count += 1
        return self.wait_count >= self.patience

    def reset(self) -> None:
        self.wait_count = 0
        self.best_loss = float("inf")


class ValidationEvaluator:
    def __init__(self, model: Model, validation_set: Dataset, loss_fn: LossFunction) -> None:
        self.model = model
        self.validation_set = validation_set
        self.loss_fn = loss_fn

    def evaluate(self) -> float:
        n = self.validation_set.size()
        if n == 0:
            return float("nan")
        feat = [[0.0] * self.validation_set.feature_dim()]
        tgt = [[0.0] * self.validation_set.target_dim()]
        out = [[0.0] * self.validation_set.target_dim()]
        total = 0.0
        for i in range(n):
            self.validation_set.get_batch(i, 1, feat, tgt)
            self.model.forward(feat, out)
            total += self.loss_fn.compute(out[0], tgt[0])
        return total / n


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------


class TP88Logger:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.lines: List[str] = []

    def log(self, level: str, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{self.run_id}] [{level}] {msg}"
        self.lines.append(line)
        print(line)

    def info(self, msg: str) -> None:
        self.log("INFO", msg)

    def warn(self, msg: str) -> None:
        self.log("WARN", msg)

    def error(self, msg: str) -> None:
        self.log("ERROR", msg)

    def write_to_file(self, path: Union[str, Path]) -> None:
        Path(path).write_text("\n".join(self.lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# RUN COMPARATOR & METRICS AGGREGATOR
# -----------------------------------------------------------------------------


class RunComparator:
    def __init__(self, registry: RunRegistry) -> None:
        self.registry = registry

    def get_best_run_by_loss(self, run_ids: List[str]) -> Optional[str]:
        if not run_ids:
            return None
        best_id = run_ids[0]
        best_loss = float("inf")
        for rid in run_ids:
            epochs = self.registry.get_epochs(rid)
            if not epochs:
                continue
            last = epochs[-1].loss
            if last < best_loss:
                best_loss = last
                best_id = rid
        return best_id

    def get_final_loss_per_run(self, run_ids: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for rid in run_ids:
            epochs = self.registry.get_epochs(rid)
            out[rid] = epochs[-1].loss if epochs else float("nan")
        return out


class MetricsAggregator:
    def __init__(self) -> None:
        self.history: List[EpochMetrics] = []

    def add(self, m: EpochMetrics) -> None:
        self.history.append(m)

    def get_best_loss(self) -> float:
        if not self.history:
            return float("inf")
        return min(m.loss for m in self.history)

    def get_last_loss(self) -> float:
        if not self.history:
            return float("nan")
        return self.history[-1].loss

    def get_history(self) -> List[EpochMetrics]:
        return list(self.history)


# -----------------------------------------------------------------------------
# CHECKPOINT MANAGER
# -----------------------------------------------------------------------------


class CheckpointManager:
    def __init__(self, base_dir: Union[str, Path], registry: RunRegistry) -> None:
        self.base_dir = Path(base_dir)
        self.registry = registry

    def save_checkpoint(
        self,
        run_id: str,
        checkpoint_index: int,
        model: Model,
        config: TrainingConfig,
    ) -> None:
        dir_path = self.base_dir / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"ckpt_{checkpoint_index}.bin"
        params = model.get_params()
        with open(file_path, "wb") as f:
            f.write(struct.pack("i", config.max_epochs))
            f.write(struct.pack("i", config.batch_size))
            f.write(struct.pack("i", len(params)))
            f.write(struct.pack(f"{len(params)}d", *params))

    def load_checkpoint(
        self,
        run_id: str,
        checkpoint_index: int,
        model: Model,
    ) -> None:
        file_path = self.base_dir / run_id / f"ckpt_{checkpoint_index}.bin"
        if not file_path.exists():
            raise TP88CheckpointError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            f.read(4)  # max_epochs
            f.read(4)  # batch_size
            n = struct.unpack("i", f.read(4))[0]
            buf = f.read(n * 8)
            params = list(struct.unpack(f"{n}d", buf))
            model.set_params(params)


# -----------------------------------------------------------------------------
# LR SCHEDULER
# -----------------------------------------------------------------------------


class LRScheduler:
    def get_lr(self, epoch: int, step: int) -> float:
        raise NotImplementedError


class StepLRScheduler(LRScheduler):
    def __init__(self, initial_lr: float, step_size: int, gamma: float) -> None:
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int, step: int) -> float:
        s = epoch * 1000 + step
        return self.initial_lr * (self.gamma ** (s // self.step_size))


class CosineAnnealingScheduler(LRScheduler):
    def __init__(self, initial_lr: float, total_steps: int) -> None:
        self.initial_lr = initial_lr
        self.total_steps = total_steps

    def get_lr(self, epoch: int, step: int) -> float:
        s = epoch * 1000 + step
        if s >= self.total_steps:
            return self.initial_lr * 0.01
        return 0.5 * self.initial_lr * (1 + math.cos(math.pi * s / self.total_steps))


# -----------------------------------------------------------------------------
# TRAINING CALLBACKS
# -----------------------------------------------------------------------------


class TrainingCallback:
    def on_epoch_start(self, run_id: str, epoch: int) -> None:
        pass

    def on_epoch_end(self, run_id: str, epoch: int, metrics: EpochMetrics) -> None:
        pass

    def on_checkpoint(self, run_id: str, checkpoint_index: int) -> None:
        pass

    def on_run_complete(self, run_id: str) -> None:
        pass


class LoggingCallback(TrainingCallback):
    def __init__(self, logger: TP88Logger) -> None:
        self.logger = logger

    def on_epoch_start(self, run_id: str, epoch: int) -> None:
        self.logger.info(f"Epoch start: {epoch}")

    def on_epoch_end(self, run_id: str, epoch: int, metrics: EpochMetrics) -> None:
        self.logger.info(str(metrics))

    def on_checkpoint(self, run_id: str, checkpoint_index: int) -> None:
        self.logger.info(f"Checkpoint: {checkpoint_index}")

    def on_run_complete(self, run_id: str) -> None:
        self.logger.info(f"Run complete: {run_id}")


class CompositeCallback(TrainingCallback):
    def __init__(self) -> None:
        self.callbacks: List[TrainingCallback] = []

    def add(self, c: TrainingCallback) -> None:
        self.callbacks.append(c)

    def on_epoch_start(self, run_id: str, epoch: int) -> None:
        for c in self.callbacks:
            c.on_epoch_start(run_id, epoch)

    def on_epoch_end(self, run_id: str, epoch: int, metrics: EpochMetrics) -> None:
        for c in self.callbacks:
            c.on_epoch_end(run_id, epoch, metrics)

    def on_checkpoint(self, run_id: str, checkpoint_index: int) -> None:
        for c in self.callbacks:
            c.on_checkpoint(run_id, checkpoint_index)

    def on_run_complete(self, run_id: str) -> None:
        for c in self.callbacks:
            c.on_run_complete(run_id)


# -----------------------------------------------------------------------------
# PROXIMA REPORTER
# -----------------------------------------------------------------------------


class ProximaReporter:
    def __init__(self, registry: RunRegistry, logger: TP88Logger) -> None:
        self.registry = registry
        self.logger = logger

    def report_run(self, run_id: str) -> None:
        r = self.registry.get_run(run_id)
        self.logger.info(f"Run {run_id} submitter={r.submitter_id} epochs={r.epoch_count}")
        epochs = self.registry.get_epochs(run_id)
        if epochs:
            first_loss = epochs[0].loss
            last_loss = epochs[-1].loss
            self.logger.info(f"  first loss={first_loss} last loss={last_loss}")
        self.logger.info(f"  checkpoints={len(self.registry.get_checkpoints(run_id))}")

    def report_all_runs(self) -> None:
        for rid in self.registry.get_all_run_ids():
            self.report_run(rid)


# -----------------------------------------------------------------------------
# MULTI-RUN RUNNER
# -----------------------------------------------------------------------------


class MultiRunRunner:
    def __init__(
        self,
        registry: RunRegistry,
        config: TrainingConfig,
        num_runs: int,
    ) -> None:
        self.registry = registry
        self.config = config
        self.num_runs = num_runs

    def run_all(
        self,
        feature_dim: int,
        target_dim: int,
        num_samples: int,
    ) -> List[str]:
        run_ids: List[str] = []
        loss_fn = create_loss(self.config.loss_name)
        for r in range(self.num_runs):
            seed = self.config.random_seed + r * 9973
            ds = generate_synthetic_random(num_samples, feature_dim, target_dim, seed)
            rng = random.Random(seed)
            model = LinearModel(feature_dim, target_dim, rng)
            opt = create_optimizer(
                self.config.optimizer_name,
                self.config.learning_rate,
                model.param_count(),
            )
            bot = TrainerBot(self.registry, self.config, loss_fn, opt, model, ds)
            run_id = bot.start_run(f"batch_submitter_{r}")
            bot.run_training(run_id)
            run_ids.append(run_id)
        return run_ids


# -----------------------------------------------------------------------------
# RUN FILTER & STATS
# -----------------------------------------------------------------------------


def filter_runs_by_submitter(registry: RunRegistry, submitter_id: str) -> List[str]:
    return [
        rid for rid in registry.get_all_run_ids()
        if registry.get_run(rid).submitter_id == submitter_id
    ]


def filter_runs_min_epochs(registry: RunRegistry, min_epochs: int) -> List[str]:
    return [
        rid for rid in registry.get_all_run_ids()
        if registry.get_run(rid).epochs_recorded >= min_epochs
    ]


def filter_runs_non_archived(registry: RunRegistry) -> List[str]:
    return [
        rid for rid in registry.get_all_run_ids()
        if not registry.get_run(rid).archived
    ]


def stats_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def stats_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = stats_mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def stats_min(values: List[float]) -> float:
    return min(values) if values else float("nan")


def stats_max(values: List[float]) -> float:
    return max(values) if values else float("nan")


# -----------------------------------------------------------------------------
# CONFIG VALIDATOR
# -----------------------------------------------------------------------------


def validate_config(c: TrainingConfig) -> None:
    if c.max_epochs <= 0:
        raise TP88ConfigValidationError("max_epochs")
    if c.batch_size <= 0:
        raise TP88ConfigValidationError("batch_size")
    if c.learning_rate <= 0 or not math.isfinite(c.learning_rate):
        raise TP88ConfigValidationError("learning_rate")
    if c.gradient_clip_norm <= 0:
        raise TP88ConfigValidationError("gradient_clip_norm")
    if c.checkpoint_every_epochs <= 0:
        raise TP88ConfigValidationError("checkpoint_every_epochs")


# -----------------------------------------------------------------------------
# RUN ID GENERATOR & HASH UTILS
# -----------------------------------------------------------------------------


class RunIdGenerator:
    def __init__(self, prefix: Optional[str] = None) -> None:
        self.prefix = prefix or TP88_RUN_PREFIX
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"{self.prefix}{self._counter:016x}{int(time.time()*1e6) % (2**24):06x}"


def config_hash_bytes(c: TrainingConfig) -> bytes:
    return hashlib.sha256(config_to_json(c).encode()).digest()


def bytes_to_hex(b: bytes) -> str:
    return b.hex()


# -----------------------------------------------------------------------------
# DATASET SPLIT
# -----------------------------------------------------------------------------


def train_val_split(
    full: ArrayDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[ArrayDataset, ArrayDataset]:
    if val_ratio <= 0 or val_ratio >= 1:
        raise TP88ConfigValidationError("val_ratio")
    n = full.size()
    val_size = int(n * val_ratio)
    train_size = n - val_size
    fd = full.feature_dim()
    td = full.target_dim()
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    train_feat = [[0.0] * fd for _ in range(train_size)]
    train_tgt = [[0.0] * td for _ in range(train_size)]
    val_feat = [[0.0] * fd for _ in range(val_size)]
    val_tgt = [[0.0] * td for _ in range(val_size)]
    for i in range(train_size):
        full.get_batch(indices[i], 1, [train_feat[i]], [train_tgt[i]])
    for i in range(val_size):
        full.get_batch(indices[train_size + i], 1, [val_feat[i]], [val_tgt[i]])
    return (
        ArrayDataset(train_feat, train_tgt, seed),
        ArrayDataset(val_feat, val_tgt, seed + 1),
    )


# -----------------------------------------------------------------------------
# EPOCH RECORD HELPERS
# -----------------------------------------------------------------------------


def epoch_record_min_loss(records: List[EpochRecord]) -> float:
    if not records:
        return float("nan")
    return min(r.loss for r in records)


def epoch_record_max_loss(records: List[EpochRecord]) -> float:
    if not records:
        return float("nan")
    return max(r.loss for r in records)


def epoch_record_avg_loss(records: List[EpochRecord]) -> float:
    if not records:
        return float("nan")
    return sum(r.loss for r in records) / len(records)


# -----------------------------------------------------------------------------
# PROXIMA RUN STATUS
# -----------------------------------------------------------------------------


class ProximaRunStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


def resolve_run_status(registry: RunRegistry, run_id: str) -> str:
    try:
        r = registry.get_run(run_id)
        if r.archived:
            return ProximaRunStatus.ARCHIVED
        recorded = r.epochs_recorded
        if recorded == 0:
            return ProximaRunStatus.PENDING
        if recorded >= r.epoch_count:
            return ProximaRunStatus.COMPLETED
        return ProximaRunStatus.RUNNING
    except TP88RunNotFoundError:
        return ProximaRunStatus.FAILED


# -----------------------------------------------------------------------------
# PROXIMA VERSION & CONSTANTS EXT
# -----------------------------------------------------------------------------


TP88_MAJOR = 8
TP88_MINOR = 8
TP88_NAME = "TP88"


def tp88_version_string() -> str:
    return f"{TP88_NAME} v{TP88_MAJOR}.{TP88_MINOR}"


TP88_DEFAULT_VAL_SPLIT_PERCENT = 20
TP88_DEFAULT_EARLY_STOP_PATIENCE = 15
TP88_DEFAULT_EARLY_STOP_MIN_DELTA = 1e-4
TP88_MAX_RUN_ID_LEN = 64


# -----------------------------------------------------------------------------
# WARMUP RUNNER & TIMING
# -----------------------------------------------------------------------------


def warmup_model(
    model: Model,
    dataset: Dataset,
    batch_size: int,
    warmup_batches: int,
) -> None:
    fd = dataset.feature_dim()
    td = dataset.target_dim()
    feat = [[0.0] * fd for _ in range(batch_size)]
    tgt = [[0.0] * td for _ in range(batch_size)]
    out = [[0.0] * td for _ in range(batch_size)]
    for b in range(warmup_batches):
        start = b * batch_size
        length = min(batch_size, dataset.size() - start)
        if length <= 0:
            break
        dataset.get_batch(start, length, feat, tgt)
        model.forward(feat, out)


class ProximaTiming:
    def __init__(self) -> None:
        self._start_ns = 0.0

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def elapsed_ns(self) -> int:
        return time.perf_counter_ns() - int(self._start_ns)

    def elapsed_ms(self) -> float:
        return self.elapsed_ns() / 1e6


# -----------------------------------------------------------------------------
# BATCH SCHEDULER & LOSS RECORD
# -----------------------------------------------------------------------------


class ProximaBatchScheduler:
    def __init__(self, total_batches: int, rng: random.Random) -> None:
        self.total_batches = total_batches
        self.order = list(range(total_batches))
        rng.shuffle(self.order)
        self._cursor = 0

    def next_batch_index(self) -> int:
        idx = self.order[self._cursor % self.total_batches]
        self._cursor += 1
        return idx

    def reset(self) -> None:
        self._cursor = 0


@dataclass
class ProximaLossRecord:
    step: int
    value: float
    timestamp_ms: float


class ProximaStepRecorder:
    def __init__(self) -> None:
        self.records: List[ProximaLossRecord] = []
