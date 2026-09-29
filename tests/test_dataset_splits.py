"""Tests for Phase 9: Dataset Splitting and Leakage Prevention.

Verifies:
- Grouped split strictly by parent_circuit_id (all candidates from the same parent circuit are in one split).
- Disjointness of parent IDs across train, validation, and test splits (no leakage).
- Verification that ratio distributions conform to configuration (70 / 15 / 15).
- Deterministic reproducibility under a fixed random seed.
- Exception raised if leakage is introduced or ratios do not sum to 1.0.
- Save and load round-trip of split datasets.
"""

import numpy as np
import pytest

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset import (
    DatasetSplits,
    split_dataset,
    split_dataset_from_config,
)
from qco.data.dataset_builder import DatasetBuilder
from qco.data.schemas import ObservationRecord


def _generate_synthetic_records(num_parents: int = 20, candidates_per_parent: int = 4):
    """Helper to generate mock observation records across multiple parents."""
    records = []
    circuit_store = {}
    for p_idx in range(num_parents):
        parent_id = f"parent_{p_idx:03d}"
        parent_circuit = CircuitData(
            num_qubits=2,
            gates=[GateOperation("H", (0,)), GateOperation("CX", (0, 1))],
            circuit_id=parent_id,
        )
        circuit_store[parent_id] = parent_circuit

        for c_idx in range(candidates_per_parent):
            cand_id = f"cand_{p_idx:03d}_{c_idx:02d}"
            cand_circuit = CircuitData(
                num_qubits=2,
                gates=[GateOperation("X", (0,))],
                circuit_id=f"c_circ_{p_idx:03d}_{c_idx:02d}",
            )
            circuit_store[cand_circuit.circuit_id] = cand_circuit

            record = ObservationRecord(
                parent_circuit_id=parent_id,
                candidate_id=cand_id,
                transformation_type="SelfInverseCancellation",
                transformation_location=f"qubit 0 at index {c_idx}",
                parent_gate_count=2,
                parent_cnot_count=1,
                parent_depth=2,
                candidate_gate_count=1,
                candidate_cnot_count=0,
                candidate_depth=1,
                gate_reduction=1,
                cnot_reduction=1,
                depth_reduction=1,
                equivalent=True,
                candidate_circuit_id=cand_circuit.circuit_id,
            )
            records.append(record)

    return records, circuit_store


def test_split_by_parent_disjointness():
    """Verify that parent IDs across train, val, and test are completely disjoint (No Leakage)."""
    records, store = _generate_synthetic_records(num_parents=40, candidates_per_parent=3)
    splits = split_dataset(records, circuit_store=store, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # 1. Check parent sets
    train_parents = splits.train_parent_ids
    val_parents = splits.val_parent_ids
    test_parents = splits.test_parent_ids

    assert len(train_parents) > 0
    assert len(val_parents) > 0
    assert len(test_parents) > 0

    # Strict disjointness
    assert train_parents.isdisjoint(val_parents), "Leakage detected between train and val!"
    assert train_parents.isdisjoint(test_parents), "Leakage detected between train and test!"
    assert val_parents.isdisjoint(test_parents), "Leakage detected between val and test!"

    # Total parents preserved
    total_parents = train_parents | val_parents | test_parents
    assert len(total_parents) == 40

    # Total records preserved
    assert len(splits.train_records) + len(splits.val_records) + len(splits.test_records) == len(records)


def test_candidates_from_same_parent_stay_together():
    """Verify all candidate rows derived from the same parent circuit stay in the exact same split."""
    records, store = _generate_synthetic_records(num_parents=20, candidates_per_parent=5)
    splits = split_dataset(records, circuit_store=store, seed=123)

    # Map each parent_circuit_id to the splits where its candidate records appear
    parent_to_splits = {}
    for r in splits.train_records:
        parent_to_splits.setdefault(r.parent_circuit_id, set()).add("train")
    for r in splits.val_records:
        parent_to_splits.setdefault(r.parent_circuit_id, set()).add("val")
    for r in splits.test_records:
        parent_to_splits.setdefault(r.parent_circuit_id, set()).add("test")

    # Every parent must appear in exactly ONE split
    for pid, s_set in parent_to_splits.items():
        assert len(s_set) == 1, f"Parent {pid} leaked across multiple splits: {s_set}"


def test_split_reproducibility():
    """Verify that a fixed random seed produces identical splits."""
    records, store = _generate_synthetic_records(num_parents=30, candidates_per_parent=2)

    splits1 = split_dataset(records, store, seed=999)
    splits2 = split_dataset(records, store, seed=999)

    assert splits1.train_parent_ids == splits2.train_parent_ids
    assert splits1.val_parent_ids == splits2.val_parent_ids
    assert splits1.test_parent_ids == splits2.test_parent_ids


def test_split_ratio_validation():
    """Verify error is raised when split ratios do not sum to 1.0."""
    records, store = _generate_synthetic_records(num_parents=10, candidates_per_parent=2)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_dataset(records, store, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)


def test_leakage_validator_detects_overlap():
    """Verify validate_leakage raises ValueError if artificially contaminated."""
    rec = ObservationRecord(
        parent_circuit_id="leaked_parent",
        candidate_id="c1",
        transformation_type="X",
        transformation_location="loc",
        parent_gate_count=1,
        parent_cnot_count=0,
        parent_depth=1,
        candidate_gate_count=1,
        candidate_cnot_count=0,
        candidate_depth=1,
        gate_reduction=0,
        cnot_reduction=0,
        depth_reduction=0,
        equivalent=True,
    )
    splits = DatasetSplits(
        train_records=[rec],
        val_records=[rec],  # Intentionally leaked
        test_records=[],
        circuit_store={},
    )
    with pytest.raises(ValueError, match="Data leakage detected"):
        splits.validate_leakage()


def test_split_dataset_from_config():
    """Verify splitting using default project configuration."""
    records, store = _generate_synthetic_records(num_parents=20, candidates_per_parent=3)
    splits = split_dataset_from_config(records, circuit_store=store)

    assert splits.validate_leakage() is True
    # Default config: 70% train, 15% val, 15% test
    # 20 parents -> floor(0.70*20) = 14 train, floor(0.15*20) = 3 val, 3 test
    assert len(splits.train_parent_ids) == 14
    assert len(splits.val_parent_ids) == 3
    assert len(splits.test_parent_ids) == 3


def test_save_and_load_splits_round_trip(tmp_path):
    """Verify split dataset saves to disk and loads back preserving all splits and circuit references."""
    records, store = _generate_synthetic_records(num_parents=15, candidates_per_parent=2)
    splits = split_dataset(records, circuit_store=store, seed=42)

    save_dir = tmp_path / "splits_data"
    splits.save(save_dir)

    assert (save_dir / "train.jsonl").exists()
    assert (save_dir / "val.jsonl").exists()
    assert (save_dir / "test.jsonl").exists()
    assert (save_dir / "split_metadata.json").exists()
    assert (save_dir / "circuits").is_dir()

    loaded = DatasetSplits.load(save_dir)
    assert len(loaded.train_records) == len(splits.train_records)
    assert len(loaded.val_records) == len(splits.val_records)
    assert len(loaded.test_records) == len(splits.test_records)
    assert loaded.train_parent_ids == splits.train_parent_ids
    assert loaded.val_parent_ids == splits.val_parent_ids
    assert loaded.test_parent_ids == splits.test_parent_ids


def test_end_to_end_generator_to_builder_to_split(tmp_path):
    """End-to-end integration: CircuitGenerator -> DatasetBuilder -> split_dataset."""
    # 1. Procedurally generate diverse circuits (Phase 3)
    gen = CircuitGenerator(min_qubits=2, max_qubits=3, min_gates=5, max_gates=12, seed=77)
    circuits = gen.generate_batch(15)

    # 2. Build verified dataset (Phases 5, 7, 8)
    builder = DatasetBuilder()
    builder.process_batch(circuits)

    if len(builder.records) > 0:
        # 3. Split with leakage prevention (Phase 9)
        splits = split_dataset_from_config(builder.records, circuit_store=builder.circuit_store)
        assert splits.validate_leakage() is True

        # 4. Save and reload
        splits.save(tmp_path / "e2e_splits")
        reloaded = DatasetSplits.load(tmp_path / "e2e_splits")
        assert len(reloaded.train_records) == len(splits.train_records)
        assert reloaded.train_parent_ids == splits.train_parent_ids
