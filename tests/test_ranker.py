"""Tests for Phase 15: GNN Candidate Ranking.

Verifies:
1. CandidateRanker scores and ranks generated candidates by predicted benefit descending.
2. Ranking preserves traceability and does NOT alter candidate circuit definitions.
3. Ranking on unseen circuits evaluates independently calculated outcomes:
   - Evaluates Spearman rank correlation.
   - Evaluates Top-1 success and Top-k success.
4. Critical rule: predictions are strictly proposals and must still pass independent Phase 5 verification.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset import split_dataset
from qco.data.dataset_builder import DatasetBuilder
from qco.data.gnn_dataset import GNNDataset, collate_gnn_batch
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.models.gnn import CircuitGNN
from qco.models.ranker import CandidateRanker, RankedCandidate, compute_ranking_metrics
from qco.models.trainer import GNNTrainer
from qco.verification.equivalence import check_equivalence


def test_ranking_metrics_calculation():
    """Verify compute_ranking_metrics on known synthetic values."""
    # Perfectly concordant predictions
    preds = [3.0, 2.0, 1.0, 0.0]
    actuals = [3.0, 2.0, 1.0, 0.0]

    metrics = compute_ranking_metrics(preds, actuals, k=2)
    assert pytest.approx(metrics["spearman_rho"], 1e-4) == 1.0
    assert metrics["top1_success"] == 1.0
    assert metrics["topk_success"] == 1.0
    assert metrics["topk_mean_actual_improvement"] == 2.5
    assert metrics["best_possible_improvement"] == 3.0

    # Inverted predictions
    preds_inv = [0.0, 1.0, 2.0, 3.0]
    metrics_inv = compute_ranking_metrics(preds_inv, actuals, k=1)
    assert pytest.approx(metrics_inv["spearman_rho"], 1e-4) == -1.0
    assert metrics_inv["top1_success"] == 0.0  # actual was 0.0


def _train_quick_model():
    """Helper to train a lightweight CircuitGNN for ranking tests."""
    gen = CircuitGenerator(min_qubits=2, max_qubits=3, min_gates=6, max_gates=16, seed=99)
    circuits = gen.generate_batch(20)

    builder = DatasetBuilder()
    builder.process_batch(circuits)

    splits = split_dataset(
        builder.records,
        circuit_store=builder.circuit_store,
        train_ratio=0.70,
        val_ratio=0.30,
        test_ratio=0.0,
        seed=42,
    )

    train_ds = GNNDataset(splits.train_records, splits.circuit_store)
    val_ds = GNNDataset(splits.val_records, splits.circuit_store)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_gnn_batch)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_gnn_batch)

    model = CircuitGNN(hidden_dim=32, num_gnn_layers=2)
    trainer = GNNTrainer(model, lr=0.005)
    trainer.fit(train_loader, val_loader, epochs=5)

    return model


def test_candidate_ranker_ordering_and_traceability():
    """Verify CandidateRanker generates ranked candidates in descending order of predicted score."""
    model = _train_quick_model()
    ranker = CandidateRanker(model)

    test_circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),         # match 1: cancels 2 gates
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),       # match 2: cancels 2 gates
            GateOperation("RX", (1,), (0.3,)),
            GateOperation("RX", (1,), (0.5,)), # match 3: combines 2 -> 1
        ],
    )

    ranked_candidates = ranker.rank_candidates(test_circuit)
    assert len(ranked_candidates) == 3

    # Check descending order
    scores = [rc.predicted_score for rc in ranked_candidates]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]

    # Check rank field assignments
    assert [rc.rank for rc in ranked_candidates] == [1, 2, 3]

    # Traceability: each candidate references parent
    for rc in ranked_candidates:
        assert rc.candidate.parent_circuit_id == test_circuit.circuit_id
        assert isinstance(rc.candidate.circuit, CircuitData)


def test_unseen_circuit_ranking_evaluation():
    """Verify candidate ranking and ranking metric evaluation on unseen test circuits."""
    model = _train_quick_model()
    ranker = CandidateRanker(model)

    # Unseen reducible circuit
    unseen = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("Z", (0,)),
            GateOperation("Z", (0,)),         # cancels
            GateOperation("RY", (1,), (0.2,)),
            GateOperation("RY", (1,), (0.4,)), # combines
            GateOperation("I", (2,)),          # removes identity
        ],
    )

    ranked = ranker.rank_candidates(unseen)
    assert len(ranked) == 3

    # Calculate actual ground-truth reductions independently
    unseen_cost = compute_circuit_cost(unseen)
    actual_reductions = []
    predicted_scores = []

    for rc in ranked:
        cand_cost = compute_circuit_cost(rc.candidate.circuit)
        actual_red = unseen_cost.total_gates - cand_cost.total_gates
        actual_reductions.append(actual_red)
        predicted_scores.append(rc.predicted_score)

    metrics = compute_ranking_metrics(predicted_scores, actual_reductions, k=2)

    assert "spearman_rho" in metrics
    assert "top1_success" in metrics
    assert "topk_success" in metrics
    # All candidates in this circuit are genuine valid reductions (> 0)
    assert metrics["top1_success"] == 1.0
    assert metrics["topk_success"] == 1.0
    assert metrics["best_possible_improvement"] > 0


def test_critical_rule_ranked_candidates_require_verification():
    """Critical rule: predictions are only proposals and must undergo independent Phase 5 verification."""
    model = _train_quick_model()
    ranker = CandidateRanker(model)

    test_circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("X", (0,)),
            GateOperation("X", (0,)),
            GateOperation("H", (1,)),
        ],
    )

    ranked = ranker.rank_candidates(test_circuit)
    top_cand = ranked[0]

    # Model says top candidate has some predicted score:
    assert isinstance(top_cand.predicted_score, float)

    # BUT we do not trust the GNN prediction as truth:
    # Must independently execute Phase 5 verification:
    equiv_res = check_equivalence(test_circuit, top_cand.candidate.circuit)
    assert equiv_res.equivalent is True
    assert equiv_res.comparison_method == "unitary"
