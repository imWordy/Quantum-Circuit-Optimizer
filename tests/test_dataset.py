"""Tests for Phase 8: Candidate Verification and Ground-Truth Dataset.

Verifies:
1. Complete ground-truth pipeline:
   Initial Circuit -> Generate Candidates -> Verify Equivalence -> Compute Costs -> Store Observation
2. Dataset schema compliance:
   All required fields present and typed properly.
3. Ground-truth fidelity / audit test:
   Sample stored observation records, load parent and candidate from disk,
   re-run independent equivalence and cost metrics, and verify they match stored values exactly.
"""

from pathlib import Path
import numpy as np
import pytest

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, GateOperation
from qco.circuits.serialization import load_circuit
from qco.data.dataset_builder import DatasetBuilder
from qco.metrics.circuit_cost import compute_circuit_cost, compute_cost_reduction
from qco.verification.equivalence import check_equivalence


def test_dataset_pipeline_and_fields(tmp_path):
    """Verify that dataset builder populates all required fields and saves correctly."""
    # Circuit with predictable identities
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("RX", (1,), (0.0,)),
        ],
    )

    builder = DatasetBuilder()
    df = builder.build_dataset([circuit], output_dir=tmp_path)

    assert len(df) == 3  # H-H, CX-CX, RX(0)
    required_cols = [
        "parent_circuit_id",
        "candidate_id",
        "transformation_type",
        "transformation_location",
        "parent_gate_count",
        "candidate_gate_count",
        "parent_cnot_count",
        "candidate_cnot_count",
        "parent_depth",
        "candidate_depth",
        "gate_reduction",
        "cnot_reduction",
        "depth_reduction",
        "equivalent",
    ]
    for col in required_cols:
        assert col in df.columns

    # All candidates from valid rules must be mathematically equivalent
    assert df["equivalent"].all()

    # Files saved on disk
    assert (tmp_path / "observations.csv").exists()
    assert (tmp_path / "circuits" / f"{circuit.circuit_id}.json").exists()



def test_ground_truth_recalculation_audit(tmp_path):
    """Phase 8 Critical Test: Reconstruct parent & candidate and verify stored ground truth matches fresh recalculations."""
    # Generate circuits containing transformations
    circuits = [
        CircuitData(
            num_qubits=2,
            gates=[
                GateOperation("X", (0,)),
                GateOperation("X", (0,)),
                GateOperation("RZ", (1,), (0.2,)),
                GateOperation("RZ", (1,), (0.3,)),
            ],
        ),
        CircuitData(
            num_qubits=3,
            gates=[
                GateOperation("CX", (0, 1)),
                GateOperation("CX", (0, 1)),
                GateOperation("H", (2,)),
                GateOperation("H", (2,)),
            ],
        ),
    ]

    builder = DatasetBuilder()
    df = builder.build_dataset(circuits, output_dir=tmp_path)
    circuits_dir = tmp_path / "circuits"

    # Audit each stored observation row
    for _, row in df.iterrows():
        parent_file = circuits_dir / f"{row['parent_circuit_id']}.json"
        cand_file = circuits_dir / f"{row['candidate_id']}.json"

        assert parent_file.exists()
        assert cand_file.exists()

        parent_circuit = load_circuit(parent_file)
        cand_circuit = load_circuit(cand_file)

        # 1. Re-run equivalence independently
        re_equiv = check_equivalence(parent_circuit, cand_circuit, tolerance=row["tolerance"])
        assert re_equiv.equivalent == row["equivalent"]

        # 2. Re-run costs independently
        p_cost = compute_circuit_cost(parent_circuit)
        c_cost = compute_circuit_cost(cand_circuit)
        re_red = compute_cost_reduction(p_cost, c_cost)

        assert p_cost.total_gates == row["parent_gate_count"]
        assert c_cost.total_gates == row["candidate_gate_count"]
        assert p_cost.cnot_count == row["parent_cnot_count"]
        assert c_cost.cnot_count == row["candidate_cnot_count"]
        assert p_cost.depth == row["parent_depth"]
        assert c_cost.depth == row["candidate_depth"]

        assert re_red.gate_reduction == row["gate_reduction"]
        assert re_red.cnot_reduction == row["cnot_reduction"]
        assert re_red.depth_reduction == row["depth_reduction"]
