"""Procedural Small Quantum Circuit Generator for QCO.

Generates diverse, valid initial quantum circuit problem instances within
configured qubit and gate-count limits.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from qiskit import QuantumCircuit

from qco.circuits.representation import CircuitData, GateOperation, SUPPORTED_GATES
from qco.utils.config import load_config
from qco.utils.logging import setup_logger

logger = setup_logger("qco.circuits.generator")


class CircuitGenerator:
    """Procedural generator for small quantum circuits."""

    def __init__(
        self,
        min_qubits: int = 2,
        max_qubits: int = 5,
        min_gates: int = 5,
        max_gates: int = 30,
        allowed_gates: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize generator with qubit, gate-count, and vocabulary limits.

        Args:
            min_qubits: Minimum number of qubits (>= 1).
            max_qubits: Maximum number of qubits (>= min_qubits).
            min_gates: Minimum number of gates (>= 1).
            max_gates: Maximum number of gates (>= min_gates).
            allowed_gates: Optional list/tuple of allowed gate names. Defaults to all supported gates.
            seed: Optional integer random seed for reproducibility.
        """
        if min_qubits < 1:
            raise ValueError(f"min_qubits must be >= 1, got {min_qubits}")
        if max_qubits < min_qubits:
            raise ValueError(f"max_qubits ({max_qubits}) < min_qubits ({min_qubits})")
        if min_gates < 1:
            raise ValueError(f"min_gates must be >= 1, got {min_gates}")
        if max_gates < min_gates:
            raise ValueError(f"max_gates ({max_gates}) < min_gates ({min_gates})")

        self.min_qubits = min_qubits
        self.max_qubits = max_qubits
        self.min_gates = min_gates
        self.max_gates = max_gates

        if allowed_gates is None:
            self.allowed_gates = list(SUPPORTED_GATES.keys())
        else:
            for g in allowed_gates:
                norm_g = "CX" if g.upper() == "CNOT" else g.upper()
                if norm_g not in SUPPORTED_GATES:
                    raise ValueError(f"Unknown gate '{g}' in allowed_gates")
            self.allowed_gates = [("CX" if g.upper() == "CNOT" else g.upper()) for g in allowed_gates]

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> CircuitGenerator:
        """Create CircuitGenerator using project configuration."""
        if config is None:
            config = load_config()

        c_cfg = config.get("circuit", {})
        seed = config.get("seed", 42)

        return cls(
            min_qubits=c_cfg.get("min_qubits", 2),
            max_qubits=c_cfg.get("max_qubits", 5),
            min_gates=c_cfg.get("min_gates", 5),
            max_gates=c_cfg.get("max_gates", 30),
            allowed_gates=c_cfg.get("allowed_gates", None),
            seed=seed,
        )

    def set_seed(self, seed: int) -> None:
        """Reset internal random generator states with a new seed."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_circuit(
        self,
        num_qubits: Optional[int] = None,
        num_gates: Optional[int] = None,
        circuit_idx: Optional[int] = None,
    ) -> CircuitData:
        """Generate a single valid quantum circuit instance.

        Args:
            num_qubits: Specific qubit count (defaults to random in [min_qubits, max_qubits]).
            num_gates: Specific gate count (defaults to random in [min_gates, max_gates]).
            circuit_idx: Optional index to record in metadata.

        Returns:
            CircuitData representing the generated circuit.
        """
        if num_qubits is None:
            num_qubits = self.rng.randint(self.min_qubits, self.max_qubits)
        if num_gates is None:
            num_gates = self.rng.randint(self.min_gates, self.max_gates)

        # In 1-qubit circuits, filter out 2-qubit gates (e.g. CX)
        available_gates = [
            g for g in self.allowed_gates if SUPPORTED_GATES[g]["num_qubits"] <= num_qubits
        ]
        if not available_gates:
            raise ValueError(f"No allowed gates compatible with {num_qubits} qubits.")

        gates: List[GateOperation] = []
        for _ in range(num_gates):
            gate_name = self.rng.choice(available_gates)
            spec = SUPPORTED_GATES[gate_name]
            needed_qubits = spec["num_qubits"]

            if needed_qubits == 1:
                target = self.rng.randint(0, num_qubits - 1)
                qubits = (target,)
            elif needed_qubits == 2:
                # Sample 2 distinct qubits for control and target
                q1, q2 = self.rng.sample(range(num_qubits), 2)
                qubits = (q1, q2)
            else:
                raise NotImplementedError(f"Gates with {needed_qubits} qubits not implemented.")

            # Sample parameters if rotation gate
            if spec["num_params"] == 1:
                # Sample rotation angle uniformly in [-pi, pi]
                angle = float(self.np_rng.uniform(-np.pi, np.pi))
                params = (angle,)
            else:
                params = ()

            gates.append(GateOperation(name=gate_name, qubits=qubits, params=params))

        metadata = {
            "generator": "CircuitGenerator",
            "target_num_qubits": num_qubits,
            "target_num_gates": num_gates,
        }
        if circuit_idx is not None:
            metadata["circuit_idx"] = circuit_idx

        return CircuitData(num_qubits=num_qubits, gates=gates, metadata=metadata)

    def generate_batch(self, count: int) -> List[CircuitData]:
        """Generate a batch of diverse circuits.

        Args:
            count: Number of circuits to generate.

        Returns:
            List of generated CircuitData instances.
        """
        circuits = []
        for i in range(count):
            circuits.append(self.generate_circuit(circuit_idx=i))
        return circuits

    def compute_distribution_stats(
        self, circuits: Sequence[CircuitData]
    ) -> Dict[str, Any]:
        """Compute statistical distribution metrics over a collection of circuits.

        Records:
        - number of circuits
        - gate-count distribution
        - depth distribution
        - CNOT distribution
        - qubit distribution
        - gate-type distribution
        """
        num_circuits = len(circuits)
        if num_circuits == 0:
            return {"num_circuits": 0}

        gate_counts = [len(c) for c in circuits]
        qubit_counts = [c.num_qubits for c in circuits]

        cnot_counts = []
        depths = []
        gate_type_counts: Dict[str, int] = {}

        for c in circuits:
            qc = c.to_qiskit()
            depths.append(qc.depth())
            cnots = sum(1 for g in c.gates if g.name in ("CX", "CNOT"))
            cnot_counts.append(cnots)

            for g in c.gates:
                gate_type_counts[g.name] = gate_type_counts.get(g.name, 0) + 1

        stats = {
            "num_circuits": num_circuits,
            "qubit_distribution": {
                "min": int(np.min(qubit_counts)),
                "max": int(np.max(qubit_counts)),
                "mean": float(np.mean(qubit_counts)),
                "std": float(np.std(qubit_counts)),
                "counts_per_qubit": {
                    int(q): int(qubit_counts.count(q)) for q in sorted(set(qubit_counts))
                },
            },
            "gate_count_distribution": {
                "min": int(np.min(gate_counts)),
                "max": int(np.max(gate_counts)),
                "mean": float(np.mean(gate_counts)),
                "std": float(np.std(gate_counts)),
            },
            "depth_distribution": {
                "min": int(np.min(depths)),
                "max": int(np.max(depths)),
                "mean": float(np.mean(depths)),
                "std": float(np.std(depths)),
            },
            "cnot_distribution": {
                "min": int(np.min(cnot_counts)),
                "max": int(np.max(cnot_counts)),
                "mean": float(np.mean(cnot_counts)),
                "std": float(np.std(cnot_counts)),
                "total_cnots": int(sum(cnot_counts)),
            },
            "gate_type_distribution": gate_type_counts,
        }

        logger.info(
            f"Generated {num_circuits} circuits: "
            f"Avg Qubits={stats['qubit_distribution']['mean']:.2f}, "
            f"Avg Gates={stats['gate_count_distribution']['mean']:.2f}, "
            f"Avg Depth={stats['depth_distribution']['mean']:.2f}, "
            f"Total CNOTs={stats['cnot_distribution']['total_cnots']}"
        )

        return stats
