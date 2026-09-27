"""Valid Transformation Rules Library for QCO.

Defines mathematically sound quantum rewrite rules with:
- Rule name
- Mathematical identity
- Applicability conditions
- Transformation function
- Expected cost effect

Rules implemented:
1. SelfInverseCancellation: X X -> I, Y Y -> I, Z Z -> I, H H -> I
2. CNOTCancellation: CX(a, b) CX(a, b) -> I
3. RotationCombination: R(a) R(b) -> R(a + b) (for RX, RY, RZ with normalized angle in (-pi, pi])
4. ZeroRotationRemoval: R(0) -> I (for RX(0), RY(0), RZ(0))
5. IdentityRemoval: I -> <empty> (clean up identity gates)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qco.circuits.representation import CircuitData, GateOperation


def normalize_angle(theta: float) -> float:
    """Normalize angle to interval (-pi, pi]."""
    # Wrap to [-pi, pi)
    angle = (theta + np.pi) % (2 * np.pi) - np.pi
    # If angle == -pi, map to +pi
    if np.isclose(angle, -np.pi, atol=1e-8):
        angle = np.pi
    # Snap close to 0 to 0.0
    if np.isclose(angle, 0.0, atol=1e-8):
        angle = 0.0
    return float(np.round(angle, 8))


@dataclass(frozen=True)
class TransformationMatch:
    """Represents an identified location in a circuit where a rule can apply."""

    rule_name: str
    indices: Tuple[int, ...]  # Gate indices in circuit.gates that participate
    qubits: Tuple[int, ...]
    description: str
    expected_gate_reduction: int
    expected_cnot_reduction: int


class TransformationRule(ABC):
    """Abstract base class for all valid quantum circuit transformation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule identifier."""
        pass

    @property
    @abstractmethod
    def mathematical_identity(self) -> str:
        """Mathematical justification."""
        pass

    @property
    @abstractmethod
    def expected_effect(self) -> str:
        """Expected effect on circuit cost."""
        pass

    @abstractmethod
    def find_matches(self, circuit: CircuitData) -> List[TransformationMatch]:
        """Find all applicable locations in the circuit for this rule."""
        pass

    @abstractmethod
    def apply(self, circuit: CircuitData, match: TransformationMatch) -> CircuitData:
        """Apply transformation at the specified match location, returning a new CircuitData."""
        pass


class SelfInverseCancellation(TransformationRule):
    """Self-inverse cancellation: U U -> I (or empty) for U in {X, Y, Z, H}."""

    SELF_INVERSE_GATES = {"X", "Y", "Z", "H"}

    @property
    def name(self) -> str:
        return "SelfInverseCancellation"

    @property
    def mathematical_identity(self) -> str:
        return "U · U = I for U ∈ {X, Y, Z, H}"

    @property
    def expected_effect(self) -> str:
        return "Removes 2 single-qubit gates (gate_reduction = 2)"

    def find_matches(self, circuit: CircuitData) -> List[TransformationMatch]:
        matches: List[TransformationMatch] = []
        gates = circuit.gates
        n = len(gates)

        # Check adjacent gates on the same qubit
        # Note: In a circuit, two gates are adjacent on qubit q if no other gate on qubit q lies between them
        for i in range(n - 1):
            g1 = gates[i]
            if g1.name in self.SELF_INVERSE_GATES and len(g1.qubits) == 1:
                q = g1.qubits[0]
                # Look forward for the immediate next gate on qubit q
                for j in range(i + 1, n):
                    g2 = gates[j]
                    if q in g2.qubits:
                        # Found the immediate successor on qubit q
                        if g2.name == g1.name and g2.qubits == g1.qubits:
                            # Verify commutativity with any intervening gates
                            # If j == i + 1, they are strictly adjacent
                            if j == i + 1:
                                matches.append(
                                    TransformationMatch(
                                        rule_name=self.name,
                                        indices=(i, j),
                                        qubits=(q,),
                                        description=f"Cancel adjacent {g1.name} gates on qubit {q} at indices ({i}, {j})",
                                        expected_gate_reduction=2,
                                        expected_cnot_reduction=0,
                                    )
                                )
                        break  # Stop checking beyond the first successor on qubit q

        return matches

    def apply(self, circuit: CircuitData, match: TransformationMatch) -> CircuitData:
        i, j = match.indices
        new_gates = [g for idx, g in enumerate(circuit.gates) if idx not in (i, j)]
        metadata = dict(circuit.metadata)
        metadata["last_transformation"] = self.name
        metadata["transformed_indices"] = list(match.indices)
        return CircuitData(num_qubits=circuit.num_qubits, gates=new_gates, metadata=metadata)


class CNOTCancellation(TransformationRule):
    """CNOT cancellation: CX(c, t) CX(c, t) -> I."""

    @property
    def name(self) -> str:
        return "CNOTCancellation"

    @property
    def mathematical_identity(self) -> str:
        return "CX(c, t) · CX(c, t) = I"

    @property
    def expected_effect(self) -> str:
        return "Removes 2 CNOT gates (gate_reduction = 2, cnot_reduction = 2)"

    def find_matches(self, circuit: CircuitData) -> List[TransformationMatch]:
        matches: List[TransformationMatch] = []
        gates = circuit.gates
        n = len(gates)

        for i in range(n - 1):
            g1 = gates[i]
            if g1.name in ("CX", "CNOT"):
                c, t = g1.qubits
                # Check if next gate is directly adjacent and identical CX
                if i + 1 < n:
                    g2 = gates[i + 1]
                    if g2.name in ("CX", "CNOT") and g2.qubits == (c, t):
                        matches.append(
                            TransformationMatch(
                                rule_name=self.name,
                                indices=(i, i + 1),
                                qubits=(c, t),
                                description=f"Cancel adjacent CX({c}, {t}) gates at indices ({i}, {i+1})",
                                expected_gate_reduction=2,
                                expected_cnot_reduction=2,
                            )
                        )
        return matches

    def apply(self, circuit: CircuitData, match: TransformationMatch) -> CircuitData:
        i, j = match.indices
        new_gates = [g for idx, g in enumerate(circuit.gates) if idx not in (i, j)]
        metadata = dict(circuit.metadata)
        metadata["last_transformation"] = self.name
        metadata["transformed_indices"] = list(match.indices)
        return CircuitData(num_qubits=circuit.num_qubits, gates=new_gates, metadata=metadata)


class RotationCombination(TransformationRule):
    """Rotation combination: R_axis(a) R_axis(b) -> R_axis(a + b)."""

    ROTATION_GATES = {"RX", "RY", "RZ"}

    @property
    def name(self) -> str:
        return "RotationCombination"

    @property
    def mathematical_identity(self) -> str:
        return "R_v(θ1) · R_v(θ2) = R_v(θ1 + θ2) for v ∈ {X, Y, Z}"

    @property
    def expected_effect(self) -> str:
        return "Merges 2 rotation gates into 1 (or 0 if angle sums to 0 mod 2pi) (gate_reduction >= 1)"

    def find_matches(self, circuit: CircuitData) -> List[TransformationMatch]:
        matches: List[TransformationMatch] = []
        gates = circuit.gates
        n = len(gates)

        for i in range(n - 1):
            g1 = gates[i]
            if g1.name in self.ROTATION_GATES:
                g2 = gates[i + 1]
                if g2.name == g1.name and g2.qubits == g1.qubits:
                    q = g1.qubits[0]
                    combined_angle = normalize_angle(g1.params[0] + g2.params[0])
                    expected_red = 2 if np.isclose(combined_angle, 0.0, atol=1e-7) else 1
                    matches.append(
                        TransformationMatch(
                            rule_name=self.name,
                            indices=(i, i + 1),
                            qubits=(q,),
                            description=(
                                f"Combine adjacent {g1.name} gates on qubit {q} at indices ({i}, {i+1}): "
                                f"({g1.params[0]:.4f} + {g2.params[0]:.4f} -> {combined_angle:.4f})"
                            ),
                            expected_gate_reduction=expected_red,
                            expected_cnot_reduction=0,
                        )
                    )
        return matches

    def apply(self, circuit: CircuitData, match: TransformationMatch) -> CircuitData:
        i, j = match.indices
        g1 = circuit.gates[i]
        g2 = circuit.gates[j]
        combined_angle = normalize_angle(g1.params[0] + g2.params[0])

        new_gates: List[GateOperation] = []
        for idx, g in enumerate(circuit.gates):
            if idx == i:
                # If sum is not 0, replace with combined gate; if 0, drop both
                if not np.isclose(combined_angle, 0.0, atol=1e-7):
                    new_gates.append(
                        GateOperation(name=g1.name, qubits=g1.qubits, params=(combined_angle,))
                    )
            elif idx == j:
                continue
            else:
                new_gates.append(g)

        metadata = dict(circuit.metadata)
        metadata["last_transformation"] = self.name
        metadata["transformed_indices"] = list(match.indices)
        return CircuitData(num_qubits=circuit.num_qubits, gates=new_gates, metadata=metadata)


class ZeroRotationRemoval(TransformationRule):
    """Zero rotation removal: R_axis(0) -> I (or empty)."""

    ROTATION_GATES = {"RX", "RY", "RZ"}

    @property
    def name(self) -> str:
        return "ZeroRotationRemoval"

    @property
    def mathematical_identity(self) -> str:
        return "R_v(0) = I for v ∈ {X, Y, Z}"

    @property
    def expected_effect(self) -> str:
        return "Removes 1 zero-rotation gate (gate_reduction = 1)"

    def find_matches(self, circuit: CircuitData) -> List[TransformationMatch]:
        matches: List[TransformationMatch] = []
        for i, g in enumerate(circuit.gates):
            if g.name in self.ROTATION_GATES and len(g.params) == 1:
                norm_angle = normalize_angle(g.params[0])
                if np.isclose(norm_angle, 0.0, atol=1e-7):
                    q = g.qubits[0]
                    matches.append(
                        TransformationMatch(
                            rule_name=self.name,
                            indices=(i,),
                            qubits=(q,),
                            description=f"Remove zero-rotation {g.name}({g.params[0]:.4f}) on qubit {q} at index {i}",
                            expected_gate_reduction=1,
                            expected_cnot_reduction=0,
                        )
                    )
        return matches

    def apply(self, circuit: CircuitData, match: TransformationMatch) -> CircuitData:
        idx_to_remove = match.indices[0]
        new_gates = [g for i, g in enumerate(circuit.gates) if i != idx_to_remove]
        metadata = dict(circuit.metadata)
        metadata["last_transformation"] = self.name
        metadata["transformed_indices"] = list(match.indices)
        return CircuitData(num_qubits=circuit.num_qubits, gates=new_gates, metadata=metadata)


class IdentityRemoval(TransformationRule):
    """Identity gate removal: I -> <empty>."""

    @property
    def name(self) -> str:
        return "IdentityRemoval"

    @property
    def mathematical_identity(self) -> str:
        return "I · U = U"

    @property
    def expected_effect(self) -> str:
        return "Removes 1 identity gate (gate_reduction = 1)"

    def find_matches(self, circuit: CircuitData) -> List[TransformationMatch]:
        matches: List[TransformationMatch] = []
        for i, g in enumerate(circuit.gates):
            if g.name == "I":
                q = g.qubits[0]
                matches.append(
                    TransformationMatch(
                        rule_name=self.name,
                        indices=(i,),
                        qubits=(q,),
                        description=f"Remove identity gate on qubit {q} at index {i}",
                        expected_gate_reduction=1,
                        expected_cnot_reduction=0,
                    )
                )
        return matches

    def apply(self, circuit: CircuitData, match: TransformationMatch) -> CircuitData:
        idx_to_remove = match.indices[0]
        new_gates = [g for i, g in enumerate(circuit.gates) if i != idx_to_remove]
        metadata = dict(circuit.metadata)
        metadata["last_transformation"] = self.name
        metadata["transformed_indices"] = list(match.indices)
        return CircuitData(num_qubits=circuit.num_qubits, gates=new_gates, metadata=metadata)


def get_default_rules() -> List[TransformationRule]:
    """Return instances of all standard valid transformation rules."""
    return [
        SelfInverseCancellation(),
        CNOTCancellation(),
        RotationCombination(),
        ZeroRotationRemoval(),
        IdentityRemoval(),
    ]
