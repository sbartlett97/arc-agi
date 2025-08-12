from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("PyTorch is required for TorchLogisticPolicy") from e
    return torch


@dataclass
class TorchLogisticPolicy:
    """Logistic regression policy implemented in PyTorch with GPU support.

    API mirrors the numpy-based LogisticPolicy for drop-in usage in scripts.
    """

    W: "object"  # torch.nn.Parameter of shape [C, D]
    b: "object"  # torch.nn.Parameter of shape [C]
    op_signatures: List[str]
    device: str = "cpu"

    @classmethod
    def init(cls, dim: int, op_signatures: List[str], device: str = "cpu") -> "TorchLogisticPolicy":
        torch = _require_torch()
        num_classes = len(op_signatures)
        W = torch.nn.Parameter(0.01 * torch.randn(num_classes, dim, device=device, dtype=torch.float32))
        b = torch.nn.Parameter(torch.zeros(num_classes, device=device, dtype=torch.float32))
        return cls(W=W, b=b, op_signatures=op_signatures, device=device)

    def predict_log_proba(self, X):
        """X: numpy array [N, D] or torch tensor, returns numpy log-probs [N, C]."""
        torch = _require_torch()
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = X @ self.W.T + self.b[None, :]
        logits = logits - logits.max(dim=1, keepdim=True).values
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return log_probs.detach().cpu().numpy()

    def fit(
        self,
        X,
        y,
        lr: float = 0.1,
        epochs: int = 5,
        l2: float = 1e-4,
        batch_size: int = 256,
    ) -> None:
        torch = _require_torch()
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long, device=self.device)

        params = [self.W, self.b]
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=l2)
        num_samples = X.shape[0]
        for _ in range(epochs):
            # simple minibatch loop
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                xb = X.index_select(0, idx)
                yb = y.index_select(0, idx)
                logits = xb @ self.W.T + self.b[None, :]
                loss = torch.nn.functional.cross_entropy(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


