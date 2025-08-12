from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .dsl import Grid
from .search import enumerate_operations


def color_histogram(grid: Grid, num_colors: int = 10) -> np.ndarray:
    hist = np.zeros((num_colors,), dtype=np.float32)
    for row in grid:
        for v in row:
            if 0 <= v < num_colors:
                hist[v] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def nonzero_bbox(grid: Grid) -> Tuple[int, int, int, int]:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    min_r, min_c = h, w
    max_r, max_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                if r < min_r:
                    min_r = r
                if c < min_c:
                    min_c = c
                if r > max_r:
                    max_r = r
                if c > max_c:
                    max_c = c
    if max_r == -1:
        return 0, 0, 0, 0
    return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1


def basic_features(inp: Grid, target: Grid, current: Grid) -> np.ndarray:
    # Concatenate histograms and simple geometry features
    hin = color_histogram(inp)
    ht = color_histogram(target)
    hc = color_histogram(current)
    hi, wi = len(inp), (len(inp[0]) if inp else 0)
    ht_h, ht_w = len(target), (len(target[0]) if target else 0)
    hc_h, hc_w = len(current), (len(current[0]) if current else 0)
    bb_in = np.array(nonzero_bbox(inp), dtype=np.float32)
    bb_t = np.array(nonzero_bbox(target), dtype=np.float32)
    bb_c = np.array(nonzero_bbox(current), dtype=np.float32)
    size_vec = np.array([hi, wi, ht_h, ht_w, hc_h, hc_w], dtype=np.float32)
    feat = np.concatenate([hin, ht, hc, size_vec, bb_in, bb_t, bb_c], axis=0)
    return feat


@dataclass
class LogisticPolicy:
    W: np.ndarray  # shape [num_classes, dim]
    b: np.ndarray  # shape [num_classes]
    op_signatures: List[str]

    @classmethod
    def init(cls, dim: int, op_signatures: List[str]) -> "LogisticPolicy":
        num_classes = len(op_signatures)
        rng = np.random.default_rng(0)
        W = 0.01 * rng.standard_normal((num_classes, dim)).astype(np.float32)
        b = np.zeros((num_classes,), dtype=np.float32)
        return cls(W=W, b=b, op_signatures=op_signatures)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        # X: [N, D]
        logits = X @ self.W.T + self.b[None, :]
        # numerical stability
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return np.log(probs + 1e-9)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 5, l2: float = 1e-4, batch_size: int = 256) -> None:
        num_samples, dim = X.shape
        num_classes = self.W.shape[0]
        for _ in range(epochs):
            # shuffle
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch = idx[start:end]
                xb = X[batch]
                yb = y[batch]
                logits = xb @ self.W.T + self.b[None, :]
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probs = exp / exp.sum(axis=1, keepdims=True)
                # gradient
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1.0
                grad_logits = (probs - onehot) / len(yb)
                grad_W = grad_logits.T @ xb + l2 * self.W
                grad_b = grad_logits.sum(axis=0)
                self.W -= lr * grad_W
                self.b -= lr * grad_b

    def save(self, path: str) -> None:
        data = {
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "op_signatures": self.op_signatures,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "LogisticPolicy":
        with open(path, "r") as f:
            data = json.load(f)
        W = np.array(data["W"], dtype=np.float32)
        b = np.array(data["b"], dtype=np.float32)
        return cls(W=W, b=b, op_signatures=list(data["op_signatures"]))


def build_imitation_dataset(tasks: Sequence[Tuple[Grid, Grid]], max_rollout_steps: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create (X, y) where y is the best single-step operation to reduce loss from input to target.

    For each pair, and for a few rollout steps, label is argmin over operations of pixel mismatch.
    """
    from .search import pixel_mismatch  # local import to avoid circular typing

    ops = enumerate_operations()
    op_sigs = [op.signature() for op in ops]
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for inp, target in tasks:
        current = inp
        for _ in range(max_rollout_steps):
            # compute label
            best_idx = 0
            best_loss = float("inf")
            for i, op in enumerate(ops):
                pred = op.apply(current)
                loss = pixel_mismatch(pred, target)
                if loss < best_loss:
                    best_loss = loss
                    best_idx = i
            feat = basic_features(inp, target, current)
            X_list.append(feat)
            y_list.append(best_idx)
            # move to next current with best op to produce rollouts
            current = ops[best_idx].apply(current)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, op_sigs


