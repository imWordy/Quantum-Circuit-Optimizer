"""Feedback records and feedback learning loop for QCO — Phase 17."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from qco.circuits.representation import CircuitData
from qco.circuits.serialization import save_circuit
from qco.data.dataset import DatasetSplits
from qco.data.gnn_dataset import GNNDataset, collate_gnn_batch
from qco.data.schemas import ObservationRecord
from qco.evaluation.metrics import compute_regression_metrics
from qco.models.gnn import CircuitGNN
from qco.models.ranker import CandidateRanker, compute_ranking_metrics
from qco.models.trainer import GNNTrainer
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer
from qco.utils.logging import setup_logger

logger = setup_logger("qco.models.feedback")


@dataclass
class FeedbackRecord:
    """Stores the predicted vs. verified actual outcome for an evaluated transformation candidate."""

    model_version: str
    circuit_id: str
    candidate_id: str
    transformation_type: str
    transformation_location: str
    predicted_benefit: float
    actual_benefit: int  # Actual gate reduction (ground truth)
    is_equivalent: bool
    candidate_gate_count: int
    candidate_cnot_count: int
    candidate_depth: int
    optimizer_iteration: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeedbackRecord:
        return cls(**data)

    def to_observation_record(self, parent_cost_gate: int, parent_cost_cnot: int, parent_cost_depth: int) -> ObservationRecord:
        """Convert verified feedback entry to an ObservationRecord for GNN dataset augmentation."""
        return ObservationRecord(
            parent_circuit_id=self.circuit_id,
            candidate_id=self.candidate_id,
            transformation_type=self.transformation_type,
            transformation_location=self.transformation_location,
            parent_gate_count=parent_cost_gate,
            parent_cnot_count=parent_cost_cnot,
            parent_depth=parent_cost_depth,
            candidate_gate_count=self.candidate_gate_count,
            candidate_cnot_count=self.candidate_cnot_count,
            candidate_depth=self.candidate_depth,
            gate_reduction=self.actual_benefit,
            cnot_reduction=parent_cost_cnot - self.candidate_cnot_count,
            depth_reduction=parent_cost_depth - self.candidate_depth,
            equivalent=self.is_equivalent,
        )


@dataclass
class FeedbackRoundResult:
    """Summary of a completed feedback learning round."""

    round_idx: int
    num_new_observations: int
    train_loss_before: float
    train_loss_after: float
    val_mae_before: float
    val_mae_after: float
    val_rmse_before: float
    val_rmse_after: float


class FeedbackLoopManager:
    """Manages collection of verified online observations and model retraining/updating."""

    def __init__(
        self,
        model: CircuitGNN,
        initial_splits: DatasetSplits,
        lr: float = 0.001,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.splits = initial_splits
        self.lr = lr
        self.device = device
        self.feedback_records: List[FeedbackRecord] = []
        self.round_history: List[FeedbackRoundResult] = []
        self.model_version_counter = 1

    @property
    def current_model_version(self) -> str:
        return f"v{self.model_version_counter}"

    def record_feedback(self, record: FeedbackRecord) -> None:
        """Accumulate an observed feedback record."""
        self.feedback_records.append(record)

    def step_feedback_round(
        self,
        new_records: List[ObservationRecord],
        new_circuits: Dict[str, CircuitData],
        epochs_per_round: int = 5,
        batch_size: int = 8,
    ) -> FeedbackRoundResult:
        """Execute one complete feedback round:

        1. Evaluate current model on validation split (baseline before update).
        2. Append verified observations to training data and circuit store.
        3. Retrain/fine-tune GNN on augmented dataset.
        4. Re-evaluate on validation split (metrics after update).
        5. Increment model version counter.

        Args:
            new_records: Newly verified observation records from optimization runs.
            new_circuits: Circuit references necessary for reconstruction.
            epochs_per_round: Number of training epochs on augmented data.
            batch_size: DataLoader batch size.

        Returns:
            FeedbackRoundResult.
        """
        round_idx = len(self.round_history) + 1
        logger.info(f"--- Starting Feedback Round {round_idx} ({len(new_records)} new observations) ---")

        # 1. Evaluate performance before update
        val_dataset_before = GNNDataset(self.splits.val_records, self.splits.circuit_store)
        val_loader = DataLoader(val_dataset_before, batch_size=batch_size, shuffle=False, collate_fn=collate_gnn_batch)

        trainer = GNNTrainer(self.model, lr=self.lr, device=self.device)
        val_loss_before, metrics_before = trainer.evaluate(val_loader)

        # 2. Append verified observations to training set
        self.splits.train_records.extend(new_records)
        self.splits.circuit_store.update(new_circuits)

        # 3. Retrain GNN on augmented dataset
        train_dataset = GNNDataset(self.splits.train_records, self.splits.circuit_store)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gnn_batch)

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_round,
        )

        # 4. Evaluate performance after update
        val_loss_after, metrics_after = trainer.evaluate(val_loader)

        self.model_version_counter += 1

        result = FeedbackRoundResult(
            round_idx=round_idx,
            num_new_observations=len(new_records),
            train_loss_before=float(history.epochs[0].train_loss) if history.epochs else 0.0,
            train_loss_after=float(history.epochs[-1].train_loss) if history.epochs else 0.0,
            val_mae_before=metrics_before["mae"],
            val_mae_after=metrics_after["mae"],
            val_rmse_before=metrics_before["rmse"],
            val_rmse_after=metrics_after["rmse"],
        )
        self.round_history.append(result)

        logger.info(
            f"Feedback Round {round_idx} Completed: "
            f"Val MAE: {result.val_mae_before:.4f} -> {result.val_mae_after:.4f}, "
            f"Val RMSE: {result.val_rmse_before:.4f} -> {result.val_rmse_after:.4f}"
        )
        return result
