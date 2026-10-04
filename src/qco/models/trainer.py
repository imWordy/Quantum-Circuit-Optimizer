"""Training Harness for GNN Candidate Benefit Regression — Phase 14.

Manages training loop, validation tracking, checkpoints, and comparative evaluation:
- Tracks training loss, validation loss, MAE, RMSE, and R2 across epochs.
- Saves best model checkpoint based on validation loss.
- Evaluates against a Trivial Mean Baseline predictor (mean training target).
- Keeps the test set completely untouched during training and checkpoint selection.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from qco.data.gnn_dataset import GNNDataset, collate_gnn_batch
from qco.evaluation.metrics import compute_regression_metrics
from qco.models.gnn import CircuitGNN
from qco.utils.logging import setup_logger

logger = setup_logger("qco.models.trainer")


@dataclass
class EpochMetrics:
    """Performance metrics logged for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_rmse: float
    val_r2: float


@dataclass
class TrainingHistory:
    """Full trace of training and validation across epochs."""

    epochs: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    trivial_baseline_val_mae: float = 0.0
    trivial_baseline_val_rmse: float = 0.0
    trivial_baseline_val_r2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": [asdict(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "trivial_baseline": {
                "val_mae": self.trivial_baseline_val_mae,
                "val_rmse": self.trivial_baseline_val_rmse,
                "val_r2": self.trivial_baseline_val_r2,
            },
        }


class GNNTrainer:
    """Trainer orchestrating GNN training, validation, and checkpointing."""

    def __init__(
        self,
        model: CircuitGNN,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.best_model_state: Optional[Dict[str, Any]] = None

    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on a DataLoader.

        Returns:
            Tuple of (mean_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model(batch)
                loss = self.criterion(preds, batch.y.view(-1, 1))

                total_loss += loss.item() * batch.num_graphs
                all_preds.append(preds.view(-1).cpu())
                all_targets.append(batch.y.view(-1).cpu())

        total_examples = sum(len(t) for t in all_targets)
        mean_loss = total_loss / max(total_examples, 1)

        y_true = torch.cat(all_targets) if all_targets else torch.empty(0)
        y_pred = torch.cat(all_preds) if all_preds else torch.empty(0)

        metrics = compute_regression_metrics(y_true, y_pred)
        return mean_loss, metrics

    def compute_trivial_baseline(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, float]:
        """Compute metrics for a trivial baseline predicting the mean training target."""
        train_targets: List[float] = []
        for batch in train_loader:
            train_targets.extend(batch.y.view(-1).tolist())

        mean_train_target = float(np.mean(train_targets)) if train_targets else 0.0

        val_targets: List[float] = []
        for batch in val_loader:
            val_targets.extend(batch.y.view(-1).tolist())

        y_true = np.array(val_targets)
        y_pred = np.full_like(y_true, fill_value=mean_train_target)

        return compute_regression_metrics(y_true, y_pred)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> TrainingHistory:
        """Run training loop across epochs, logging metrics and maintaining best checkpoint.

        Args:
            train_loader: DataLoader over training dataset.
            val_loader: DataLoader over validation dataset.
            epochs: Number of epochs to train.
            checkpoint_dir: Optional path to save best model checkpoint.

        Returns:
            TrainingHistory instance.
        """
        history = TrainingHistory()

        # Compute baseline on validation set before training
        base_metrics = self.compute_trivial_baseline(train_loader, val_loader)
        history.trivial_baseline_val_mae = base_metrics["mae"]
        history.trivial_baseline_val_rmse = base_metrics["rmse"]
        history.trivial_baseline_val_r2 = base_metrics["r2"]

        logger.info(
            f"Trivial Baseline Val MAE: {base_metrics['mae']:.4f}, "
            f"RMSE: {base_metrics['rmse']:.4f}, R2: {base_metrics['r2']:.4f}"
        )

        for ep in range(1, epochs + 1):
            # 1. Training step
            self.model.train()
            train_loss_total = 0.0
            num_train_graphs = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                preds = self.model(batch)
                loss = self.criterion(preds, batch.y.view(-1, 1))
                loss.backward()

                self.optimizer.step()
                train_loss_total += loss.item() * batch.num_graphs
                num_train_graphs += batch.num_graphs

            train_loss = train_loss_total / max(num_train_graphs, 1)

            # 2. Validation step
            val_loss, val_metrics = self.evaluate(val_loader)

            epoch_record = EpochMetrics(
                epoch=ep,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=val_metrics["mae"],
                val_rmse=val_metrics["rmse"],
                val_r2=val_metrics["r2"],
            )
            history.epochs.append(epoch_record)

            # Checkpoint tracking
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = ep
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                if checkpoint_dir is not None:
                    ckpt_path = Path(checkpoint_dir)
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state_dict": self.best_model_state,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_metrics": val_metrics,
                        },
                        ckpt_path / "best_model.pt",
                    )

            if ep % 5 == 0 or ep == epochs:
                logger.info(
                    f"Epoch {ep:02d}/{epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val MAE: {val_metrics['mae']:.4f} | "
                    f"Val R2: {val_metrics['r2']:.4f}"
                )

        # Restore best model state weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history
