"""Tests for Phase 8: Candidate Verification and Ground-Truth Dataset.

Tests the full pipeline:
1. Process a circuit through DatasetBuilder.
2. Verify records are created with correct fields.
3. Ground-truth principle: re-run equivalence and metrics independently,
   compare to stored values.
4. Dataset save and load round-trip.
5. Only equivalent candidates have ground-truth label equivalent=True.
6. Batch processing accumulates records correctly.
"""

import json
import random

import numpy as np
import pandas as pd
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset_builder import DatasetBuilder
from qco.data.schemas import ObservationRecord
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.verification.equivalence import check_equivalence


def _make_reducible_circuit() -> CircuitData:
    """Circuit with known reducible patterns for deterministic testing."""
    return CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),         # SelfInverseCancellation
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),       # CNOTCancellation
            GateOperation("RX", (2,), (0.4,)),
            GateOperation("RX", (2,), (0.6,)), # RotationCombination
            GateOperation("RY", (1,), (0.0,)), # ZeroRotationRemoval
            GateOperation("I", (2,)),          # IdentityRemoval
        ],
    )


def test_dataset_builder_produces_records():
    """Verify DatasetBuilder processes a circuit and produces records."""
    circuit = _make_reducible_circuit()
    builder = DatasetBuilder()
    records = builder.process_circuit(circuit)

    # Expect 5 records (one per rule match)
    assert len(records) == 5
    assert len(builder.records) == 5


def test_records_have_required_fields():
    """Verify every record has all required Phase 8 dataset fields."""
    circuit = _make_reducible_circuit()
    builder = DatasetBuilder()
    records = builder.process_circuit(circuit)

    required_fields = {
        "parent_circuit_id",
        "candidate_id",
        "transformation_type",
        "transformation_location",
        "parent_gate_count",
        "parent_cnot_count",
        "parent_depth",
        "candidate_gate_count",
        "candidate_cnot_count",
        "candidate_depth",
        "gate_reduction",
        "cnot_reduction",
        "depth_reduction",
        "equivalent",
        "candidate_circuit_id",
    }
    for record in records:
        record_dict = record.to_dict()
        for field in required_fields:
            assert field in record_dict, f"Missing field: {field}"


def test_ground_truth_reproducibility():
    """Ground-truth principle: randomly inspect records and re-verify independently.

    Steps per record:
    1. Reconstruct parent from circuit_store.
    2. Reconstruct candidate from circuit_store.
    3. Re-run equivalence check.
    4. Re-run cost metrics.
    5. Assert all stored values match recomputed values.
    """
    circuit = _make_reducible_circuit()
    builder = DatasetBuilder()
    records = builder.process_circuit(circuit)

    store = builder.circuit_store

    # Check all records (for this small set; spec says "random sample")
    for record in records:
        parent = store[record.parent_circuit_id]
        candidate = store[record.candidate_circuit_id]

        # Re-run Phase 5 equivalence
        equiv = check_equivalence(parent, candidate)
        assert equiv.equivalent == record.equivalent, (
            f"Equivalence mismatch for {record.candidate_id}: "
            f"stored={record.equivalent}, recomputed={equiv.equivalent}"
        )

        # Re-run Phase 4 cost metrics
        parent_cost = compute_circuit_cost(parent)
        cand_cost = compute_circuit_cost(candidate)

        assert parent_cost.total_gates == record.parent_gate_count
        assert parent_cost.cnot_count == record.parent_cnot_count
        assert parent_cost.depth == record.parent_depth

        assert cand_cost.total_gates == record.candidate_gate_count
        assert cand_cost.cnot_count == record.candidate_cnot_count
        assert cand_cost.depth == record.candidate_depth

        assert (parent_cost.total_gates - cand_cost.total_gates) == record.gate_reduction
        assert (parent_cost.cnot_count - cand_cost.cnot_count) == record.cnot_reduction
        assert (parent_cost.depth - cand_cost.depth) == record.depth_reduction


def test_all_candidates_are_equivalent():
    """Verify all transformation-rule-generated candidates are equivalent to parent.

    All five rules produce semantically correct transformations, so equivalent=True for all.
    """
    circuit = _make_reducible_circuit()
    builder = DatasetBuilder()
    records = builder.process_circuit(circuit)

    for record in records:
        assert record.equivalent is True, (
            f"Transformation '{record.transformation_type}' produced a non-equivalent candidate!"
        )


def test_to_dataframe():
    """Verify records convert correctly to a pandas DataFrame."""
    circuit = _make_reducible_circuit()
    builder = DatasetBuilder()
    builder.process_circuit(circuit)

    df = builder.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "parent_circuit_id" in df.columns
    assert "equivalent" in df.columns
    assert "gate_reduction" in df.columns


def test_save_and_load_round_trip(tmp_path):
    """Verify dataset can be saved to disk and loaded back with matching records."""
    circuit = _make_reducible_circuit()
    builder = DatasetBuilder()
    builder.process_circuit(circuit)

    builder.save(tmp_path)

    # Verify files exist
    assert (tmp_path / "observations.jsonl").exists()
    assert (tmp_path / "observations.csv").exists()
    assert (tmp_path / "circuits").is_dir()

    # Load back
    loaded_records, loaded_circuits = DatasetBuilder.load(tmp_path)

    assert len(loaded_records) == len(builder.records)
    for orig, loaded in zip(builder.records, loaded_records):
        assert orig.parent_circuit_id == loaded.parent_circuit_id
        assert orig.equivalent == loaded.equivalent
        assert orig.gate_reduction == loaded.gate_reduction


def test_batch_processing_accumulates_records():
    """Verify processing a batch of circuits accumulates all records correctly."""
    c1 = _make_reducible_circuit()
    c2 = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("Z", (0,)),
            GateOperation("Z", (0,)),
            GateOperation("RZ", (1,), (0.5,)),
            GateOperation("RZ", (1,), (0.5,)),
        ],
    )

    builder = DatasetBuilder()
    builder.process_batch([c1, c2])

    # c1 should give 5 records, c2 should give 2 records (ZZ + RZ+RZ)
    assert len(builder.records) == 7

    parent_ids = {r.parent_circuit_id for r in builder.records}
    assert c1.circuit_id in parent_ids
    assert c2.circuit_id in parent_ids
