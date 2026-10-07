"""Tests for Phase 17: Feedback Learning.

Verifies:
1. FeedbackRecord accurately stores model version, predicted benefit, and verified actual benefit.
2. Online feedback collection works during GNNGuidedOptimizer runs.
3. FeedbackLoopManager successfully steps multiple feedback rounds:
   - Evaluates performance before update.
   - Appends verified observations to training dataset.
   - Retrains/updates GNN model without breaking or crashing.
   - Evaluates performance after update.
4. Model version increments across successive feedback rounds.
5. System remains fully stable and reproducible under feedback learning.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset import DatasetSplits, split_dataset
from qco.data.dataset_builder import DatasetBuilder
from qco.models.feedback import FeedbackLoopManager, FeedbackRecord, FeedbackRoundResult
from qco.models.gnn import CircuitGNN
from qco.models.trainer import GNNTrainer
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer


def _setup_feedback_environment():
    """Build a baseline training dataset and splits for feedback learning."""
    gen = CircuitGenerator(min_qubits=2, max_qubits=3, min_gates=6, max_gates=14, seed=42)
    circuits = gen.generate_batch(20)

    builder = DatasetBuilder()
    builder.process_batch(circuits)

    splits = split_dataset(
        builder.records,
        circuit_store=builder.circuit_store,
        train_ratio=0.70,
        val_ratio=0.30,
        test_ratio=0.0,
        seed=101,
    )

    model = CircuitGNN(hidden_dim=32, num_gnn_layers=2)
    return model, splits, gen


def test_feedback_record_creation_and_conversion():
    """Verify FeedbackRecord stores predictions, ground-truth outcomes, and converts to ObservationRecord."""
    fb = FeedbackRecord(
        model_version="v1",
        circuit_id="parent_circ_1",
        candidate_id="cand_1",
        transformation_type="SelfInverseCancellation",
        transformation_location="qubit 0 at indices (0, 1)",
        predicted_benefit=2.45,
        actual_benefit=2,
        is_equivalent=True,
        candidate_gate_count=4,
        candidate_cnot_count=1,
        candidate_depth=3,
        optimizer_iteration=1,
    )

    d = fb.to_dict()
    assert d["model_version"] == "v1"
    assert d["predicted_benefit"] == 2.45
    assert d["actual_benefit"] == 2
    assert d["is_equivalent"] is True

    # Convert to ObservationRecord
    obs = fb.to_observation_record(parent_cost_gate=6, parent_cost_cnot=1, parent_cost_depth=5)
    assert obs.parent_circuit_id == "parent_circ_1"
    assert obs.candidate_id == "cand_1"
    assert obs.gate_reduction == 2
    assert obs.equivalent is True


def test_online_feedback_collection_during_optimization():
    """Verify GNNGuidedOptimizer logs verified feedback records when collector is attached."""
    model, splits, _ = _setup_feedback_environment()
    manager = FeedbackLoopManager(model=model, initial_splits=splits)

    optimizer = GNNGuidedOptimizer(
        model=model,
        max_iterations=5,
        max_evaluations_per_iter=5,
        feedback_collector=manager,
    )

    test_circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("X", (0,)),
        ],
    )

    res = optimizer.optimize(test_circuit)
    assert res.is_equivalent is True

    # Verify feedback was accumulated in manager
    assert len(manager.feedback_records) > 0
    record = manager.feedback_records[0]
    assert record.model_version == "v1"
    assert record.circuit_id == test_circuit.circuit_id
    assert record.is_equivalent is True
    assert isinstance(record.predicted_benefit, float)
    assert isinstance(record.actual_benefit, int)


def test_feedback_learning_loop_multiple_rounds():
    """Verify FeedbackLoopManager executes multiple feedback update rounds stably."""
    model, splits, gen = _setup_feedback_environment()
    manager = FeedbackLoopManager(model=model, initial_splits=splits, lr=0.005)

    initial_train_count = len(splits.train_records)

    # 1. Round 1 update: Generate new verified circuits
    new_circuits_1 = gen.generate_batch(5)
    builder1 = DatasetBuilder()
    builder1.process_batch(new_circuits_1)

    res1 = manager.step_feedback_round(
        new_records=builder1.records,
        new_circuits=builder1.circuit_store,
        epochs_per_round=3,
        batch_size=4,
    )

    assert res1.round_idx == 1
    assert res1.num_new_observations == len(builder1.records)
    assert len(splits.train_records) == initial_train_count + len(builder1.records)
    assert manager.current_model_version == "v2"

    # 2. Round 2 update: Add another batch of verified observations
    new_circuits_2 = gen.generate_batch(5)
    builder2 = DatasetBuilder()
    builder2.process_batch(new_circuits_2)

    res2 = manager.step_feedback_round(
        new_records=builder2.records,
        new_circuits=builder2.circuit_store,
        epochs_per_round=3,
        batch_size=4,
    )

    assert res2.round_idx == 2
    assert len(manager.round_history) == 2
    assert manager.current_model_version == "v3"
    assert len(splits.train_records) == initial_train_count + len(builder1.records) + len(builder2.records)

    # System successfully incorporated verified data without crashing or producing NaNs
    assert not torch.isnan(torch.tensor(res2.val_mae_after))
    assert not torch.isnan(torch.tensor(res2.val_rmse_after))
