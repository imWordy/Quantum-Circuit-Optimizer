"""Canonical Quantum Circuit Representation for QCO.

Defines the project's source-of-truth representation of a quantum circuit,
preserving:
- number of qubits
- gate order
- gate type
- control / target qubits
- gate continuous parameters
- deterministic circuit ID
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

# Standard canonical gate vocabulary
SUPPORTED_GATES = {
    "I": {"num_qubits": 1, "num_params": 0},
    "X": {"num_qubits": 1, "num_params": 0},
    "Y": {"num_qubits": 1, "num_params": 0},
    "Z": {"num_qubits": 1, "num_params": 0},
    "H": {"num_qubits": 1, "num_params": 0},
    "RX": {"num_qubits": 1, "num_params": 1},
    "RY": {"num_qubits": 1, "num_params": 1},
    "RZ": {"num_qubits": 1, "num_params": 1},
    "CX": {"num_qubits": 2, "num_params": 0},  # control=qubits[0], target=qubits[1]
}


@dataclass(frozen=True)
class GateOperation:
    """Canonical representation of an individual gate operation."""

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        gate_name = self.name.upper()
        # Canonicalize CX / CNOT
        if gate_name == "CNOT":
            gate_name = "CX"
        object.__setattr__(self, "name", gate_name)

        if self.name not in SUPPORTED_GATES:
            raise ValueError(
                f"Unsupported gate '{self.name}'. Allowed gates: {list(SUPPORTED_GATES.keys())}"
            )

        spec = SUPPORTED_GATES[self.name]
        if len(self.qubits) != spec["num_qubits"]:
            raise ValueError(
                f"Gate '{self.name}' expects {spec['num_qubits']} qubits, but got {len(self.qubits)}: {self.qubits}"
            )

        if len(self.qubits) == 2 and self.qubits[0] == self.qubits[1]:
            raise ValueError(f"Control and target qubits cannot be identical: {self.qubits}")

        if len(self.params) != spec["num_params"]:
            raise ValueError(
                f"Gate '{self.name}' expects {spec['num_params']} parameters, but got {len(self.params)}: {self.params}"
            )

        # Ensure parameters are rounded float tuples for hashing and stability
        rounded_params = tuple(float(np.round(p, 8)) for p in self.params)
        object.__setattr__(self, "params", rounded_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert GateOperation to serializable dict."""
        return {
            "name": self.name,
            "qubits": list(self.qubits),
            "params": list(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GateOperation:
        """Create GateOperation from dict."""
        return cls(
            name=data["name"],
            qubits=tuple(data["qubits"]),
            params=tuple(data.get("params", ())),
        )


class CircuitData:
    """Canonical circuit data model and wrapper.

    Acts as the source-of-truth representation for circuits in QCO.
    """

    def __init__(
        self,
        num_qubits: int,
        gates: Optional[List[GateOperation]] = None,
        circuit_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"Circuit must have at least 1 qubit, got {num_qubits}")

        self._num_qubits = num_qubits
        self._gates: List[GateOperation] = list(gates) if gates is not None else []
        self._metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}

        # Validate gates against qubit count
        self._validate()

        # Compute or verify deterministic circuit ID
        computed_id = self.compute_id()
        self._circuit_id = circuit_id if circuit_id is not None else computed_id

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def gates(self) -> List[GateOperation]:
        return list(self._gates)

    @property
    def circuit_id(self) -> str:
        return self._circuit_id

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def __len__(self) -> int:
        """Number of gates in the circuit."""
        return len(self._gates)

    def __getitem__(self, index: int) -> GateOperation:
        return self._gates[index]

    def _validate(self) -> None:
        """Ensure all gates operate on valid qubits within [0, num_qubits - 1]."""
        for i, g in enumerate(self._gates):
            for q in g.qubits:
                if q < 0 or q >= self._num_qubits:
                    raise ValueError(
                        f"Gate #{i} ({g.name}) acts on qubit {q}, which is outside valid range [0, {self._num_qubits - 1}]"
                    )

    def compute_id(self) -> str:
        """Compute SHA256 deterministic hash ID representing the canonical circuit structure."""
        canonical_str = f"qubits:{self._num_qubits};"
        gate_strs = [
            f"{g.name}({','.join(str(q) for q in g.qubits)})"
            + (f"[{','.join(f'{p:.8f}' for p in g.params)}]" if g.params else "")
            for g in self._gates
        ]
        canonical_str += ";".join(gate_strs)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]

    def copy(self) -> CircuitData:
        """Return a deep copy of CircuitData."""
        return CircuitData(
            num_qubits=self._num_qubits,
            gates=list(self._gates),
            circuit_id=self._circuit_id,
            metadata=dict(self._metadata),
        )

    def to_qiskit(self) -> QuantumCircuit:
        """Convert canonical CircuitData to a Qiskit QuantumCircuit."""
        qc = QuantumCircuit(self._num_qubits)
        for g in self._gates:
            name = g.name.upper()
            if name == "I":
                qc.id(g.qubits[0])
            elif name == "X":
                qc.x(g.qubits[0])
            elif name == "Y":
                qc.y(g.qubits[0])
            elif name == "Z":
                qc.z(g.qubits[0])
            elif name == "H":
                qc.h(g.qubits[0])
            elif name == "RX":
                qc.rx(g.params[0], g.qubits[0])
            elif name == "RY":
                qc.ry(g.params[0], g.qubits[0])
            elif name == "RZ":
                qc.rz(g.params[0], g.qubits[0])
            elif name in ("CX", "CNOT"):
                qc.cx(g.qubits[0], g.qubits[1])
            else:
                raise ValueError(f"Cannot convert unsupported gate '{name}' to Qiskit")
        return qc

    @classmethod
    def from_qiskit(
        cls,
        qc: QuantumCircuit,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CircuitData:
        """Construct canonical CircuitData from a Qiskit QuantumCircuit.

        Ignores barriers and measurement operations.
        """
        gates: List[GateOperation] = []
        num_qubits = qc.num_qubits

        for instruction in qc.data:
            op = instruction.operation
            op_name = op.name.upper()

            # Skip barriers and measurements as they are not unitary gates
            if op_name in ("BARRIER", "MEASURE", "RESET"):
                continue

            if op_name == "ID":
                op_name = "I"
            elif op_name == "CNOT":
                op_name = "CX"

            qubits = tuple(qc.find_bit(q).index for q in instruction.qubits)
            params = tuple(float(p) for p in op.params)

            gates.append(GateOperation(name=op_name, qubits=qubits, params=params))

        return cls(num_qubits=num_qubits, gates=gates, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CircuitData to dictionary."""
        return {
            "circuit_id": self._circuit_id,
            "num_qubits": self._num_qubits,
            "gates": [g.to_dict() for g in self._gates],
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CircuitData:
        """Construct CircuitData from dictionary."""
        gates = [GateOperation.from_dict(g) for g in data.get("gates", [])]
        return cls(
            num_qubits=data["num_qubits"],
            gates=gates,
            circuit_id=data.get("circuit_id"),
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitData):
            return False
        return (
            self._num_qubits == other._num_qubits
            and self._gates == other._gates
        )

    def __repr__(self) -> str:
        return (
            f"CircuitData(id={self._circuit_id}, qubits={self._num_qubits}, "
            f"gates={len(self._gates)})"
        )
