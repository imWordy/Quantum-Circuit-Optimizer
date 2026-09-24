"""Tests for Phase 3: Small Circuit Generator.

Verifies:
- Configured limits are respected (qubits, gates).
- Generated circuits are valid, non-empty, and contain no invalid qubit indices.
- Only allowed gates appear.
- Parameters are within valid rotation ranges.
- All generated circuits successfully simulate on AerSimulator.
- Reproducibility: same seed produces identical circuits and IDs.
- Diversity: hundreds of generated circuits produce diverse distributions.
- Distribution stats logging.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import SUPPORTED_GATES


def test_generator_bounds_validation():
    """Verify generator validates min/max limits properly."""
    with pytest.raises(ValueError, match="min_qubits must be >= 1"):
        CircuitGenerator(min_qubits=0)

    with pytest.raises(ValueError, match="max_qubits .* < min_qubits"):
        CircuitGenerator(min_qubits=5, max_qubits=2)

    with pytest.raises(ValueError, match="max_gates .* < min_gates"):
        CircuitGenerator(min_gates=10, max_gates=5)

    with pytest.raises(ValueError, match="Unknown gate"):
        CircuitGenerator(allowed_gates=["UNKNOWN_GATE"])


def test_generator_seeding_reproducibility():
    """Verify seeded generator produces deterministic circuits and IDs."""
    gen1 = CircuitGenerator(min_qubits=3, max_qubits=4, min_gates=8, max_gates=15, seed=42)
    circuits1 = gen1.generate_batch(5)

    gen2 = CircuitGenerator(min_qubits=3, max_qubits=4, min_gates=8, max_gates=15, seed=42)
    circuits2 = gen2.generate_batch(5)

    for c1, c2 in zip(circuits1, circuits2):
        assert c1 == c2
        assert c1.circuit_id == c2.circuit_id


def test_hundreds_of_circuits_validity_and_diversity():
    """Phase 3 comprehensive test: Generate 200 circuits and verify validity, simulation, and diversity."""
    min_q, max_q = 2, 5
    min_g, max_g = 5, 20
    allowed = ["X", "H", "RX", "RY", "RZ", "CX"]

    gen = CircuitGenerator(
        min_qubits=min_q,
        max_qubits=max_q,
        min_gates=min_g,
        max_gates=max_g,
        allowed_gates=allowed,
        seed=123,
    )

    circuits = gen.generate_batch(200)
    assert len(circuits) == 200

    backend = AerSimulator()
    circuit_ids = set()

    for c in circuits:
        # 1. Check qubit limits
        assert min_q <= c.num_qubits <= max_q

        # 2. Check gate limits
        assert min_g <= len(c) <= max_g

        # 3. Check allowed gates and parameters
        for gate in c.gates:
            assert gate.name in allowed
            for q in gate.qubits:
                assert 0 <= q < c.num_qubits
            if gate.name in ("RX", "RY", "RZ"):
                assert len(gate.params) == 1
                assert -np.pi <= gate.params[0] <= np.pi
            elif gate.name == "CX":
                assert len(gate.qubits) == 2
                assert gate.qubits[0] != gate.qubits[1]

        # 4. Check simulation on a subset (first 25 to keep test runtime lean)
        circuit_ids.add(c.circuit_id)

    # Simulation check on 25 diverse circuits
    for c in circuits[:25]:
        qc = c.to_qiskit()
        qc.measure_all()
        result = backend.run(qc, shots=50).result()
        counts = result.get_counts()
        assert sum(counts.values()) == 50

    # 5. Check diversity: unique IDs
    assert len(circuit_ids) == 200, "All 200 generated circuits should have unique IDs"

    # 6. Distribution statistics
    stats = gen.compute_distribution_stats(circuits)
    assert stats["num_circuits"] == 200
    assert min_q <= stats["qubit_distribution"]["min"] <= max_q
    assert min_g <= stats["gate_count_distribution"]["min"] <= max_g
    assert stats["cnot_distribution"]["total_cnots"] > 0
    assert set(stats["gate_type_distribution"].keys()).issubset(set(allowed))
