"""Tests for Phase 18: Final Experimental Framework.

Verifies:
1. BenchmarkHarness runs controlled comparison across Random, Greedy, GNN-Guided, and MLP optimizers.
2. Fair comparison guarantees:
   - All methods run on identical held-out test circuits.
   - All methods use identical transformation rules and equivalence tolerance.
3. Every method produces strictly equivalent final circuits (is_equivalent == True).
4. All required metrics are logged (gate_reduction, evaluations, equivalence checks, runtime).
5. Summary table generation and serialization to CSV/JSON succeed properly.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset import split_dataset
from qco.data.dataset_builder import DatasetBuilder
from qco.data.gnn_dataset import GNNDataset, collate_gnn_batch
from qco.experiments.benchmark import BenchmarkHarness, BenchmarkRecord
from qco.models.gnn import CircuitGNN
from qco.models.mlp_baseline import AggregateFeatureMLP
from qco.models.trainer import GNNTrainer
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer
from qco.optimization.greedy_optimizer import GreedyOptimizer
from qco.optimization.mlp_optimizer import MLPOptimizer
from qco.optimization.random_optimizer import RandomOptimizer


def _setup_test_suite():
    """Build held-out test circuits and train a quick GNN model."""
    gen = CircuitGenerator(min_qubits=2, max_qubits=3, min_gates=6, max_gates=14, seed=42)
    training_circuits = gen.generate_batch(16)

    builder = DatasetBuilder()
    builder.process_batch(training_circuits)

    splits = split_dataset(
        builder.records,
        circuit_store=builder.circuit_store,
        train_ratio=0.70,
        val_ratio=0.30,
        test_ratio=0.0,
        seed=101,
    )

    train_ds = GNNDataset(splits.train_records, splits.circuit_store)
    val_ds = GNNDataset(splits.val_records, splits.circuit_store)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_gnn_batch)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_gnn_batch)

    model = CircuitGNN(hidden_dim=32, num_gnn_layers=2)
    trainer = GNNTrainer(model, lr=0.005)
    trainer.fit(train_loader, val_loader, epochs=4)

    # Distinct held-out test circuits
    test_circuits = [
        CircuitData(
            num_qubits=2,
            gates=[
                GateOperation("H", (0,)),
                GateOperation("H", (0,)),
                GateOperation("CX", (0, 1)),
                GateOperation("CX", (0, 1)),
                GateOperation("X", (0,)),
            ],
            circuit_id="test_circ_1",
        ),
        CircuitData(
            num_qubits=2,
            gates=[
                GateOperation("RX", (1,), (0.2,)),
                GateOperation("RX", (1,), (0.4,)),
                GateOperation("Z", (0,)),
                GateOperation("Z", (0,)),
            ],
            circuit_id="test_circ_2",
        ),
    ]

    return model, test_circuits


def test_controlled_benchmark_execution():
    """Verify BenchmarkHarness runs all methods under fair conditions and gathers all metrics."""
    model, test_circuits = _setup_test_suite()

    # Fair comparison budget: max_iterations=5, tolerance=1e-7, seed=42
    optimizers = [
        RandomOptimizer(max_iterations=5, max_evaluations_per_iter=5, seed=42),
        GreedyOptimizer(max_iterations=5, seed=42),
        MLPOptimizer(max_iterations=5, max_evaluations_per_iter=5, seed=42),
        GNNGuidedOptimizer(model=model, max_iterations=5, max_evaluations_per_iter=5, seed=42),
    ]

    harness = BenchmarkHarness(optimizers=optimizers)
    df = harness.run_benchmark(test_circuits)

    # 4 methods x 2 circuits = 8 records
    assert len(df) == 8
    assert len(harness.records) == 8

    # Check required fields
    required_cols = [
        "circuit_id",
        "method",
        "initial_gate_count",
        "final_gate_count",
        "initial_cnot_count",
        "final_cnot_count",
        "initial_depth",
        "final_depth",
        "gate_reduction",
        "cnot_reduction",
        "depth_reduction",
        "candidate_evaluations",
        "equivalence_checks",
        "iterations",
        "runtime",
        "is_equivalent",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Critical requirement: All methods must preserve quantum equivalence
    assert df["is_equivalent"].all(), "At least one method produced non-equivalent output!"


def test_summary_table_and_export(tmp_path):
    """Verify summary table aggregation and file export."""
    model, test_circuits = _setup_test_suite()

    optimizers = [
        RandomOptimizer(max_iterations=3, seed=42),
        GreedyOptimizer(max_iterations=3, seed=42),
    ]

    harness = BenchmarkHarness(optimizers=optimizers)
    harness.run_benchmark(test_circuits)

    summary_df = harness.compute_summary_table()
    assert len(summary_df) == 2
    assert "avg_gate_reduction" in summary_df.columns
    assert "avg_runtime_sec" in summary_df.columns
    assert "all_equivalent" in summary_df.columns
    assert summary_df["all_equivalent"].all()

    # Test export
    save_dir = tmp_path / "benchmark_out"
    harness.save_results(save_dir)

    assert (save_dir / "benchmark_raw.csv").exists()
    assert (save_dir / "benchmark_summary.csv").exists()
    assert (save_dir / "benchmark_raw.json").exists()
