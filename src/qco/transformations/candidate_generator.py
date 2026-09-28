"""Procedural Candidate Generator for QCO.

Generates candidate transformed circuits by applying valid rewrite rules from the
transformation library. It does not select or rank candidates; it strictly produces
proposals with traceable metadata without mutating the parent circuit.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import CircuitCost, compute_circuit_cost
from qco.transformations.rules import (
    TransformationMatch,
    TransformationRule,
    get_default_rules,
)


@dataclass(frozen=True)
class CandidateCircuit:
    """Represents a proposed transformed circuit with complete metadata."""

    candidate_id: str
    parent_circuit_id: str
    transformation_name: str
    transformation_location: Tuple[int, ...]
    circuit: CircuitData
    preliminary_cost: CircuitCost
    verified: bool = False
    verification_status: Optional[str] = "unverified"
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidateCircuit to serializable dict."""
        return {
            "candidate_id": self.candidate_id,
            "parent_circuit_id": self.parent_circuit_id,
            "transformation_name": self.transformation_name,
            "transformation_location": list(self.transformation_location),
            "circuit": self.circuit.to_dict(),
            "preliminary_cost": self.preliminary_cost.to_dict(),
            "verified": self.verified,
            "verification_status": self.verification_status,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateCircuit:
        """Construct CandidateCircuit from dict."""
        return cls(
            candidate_id=data["candidate_id"],
            parent_circuit_id=data["parent_circuit_id"],
            transformation_name=data["transformation_name"],
            transformation_location=tuple(data["transformation_location"]),
            circuit=CircuitData.from_dict(data["circuit"]),
            preliminary_cost=CircuitCost(**data["preliminary_cost"]),
            verified=data.get("verified", False),
            verification_status=data.get("verification_status", "unverified"),
            seed=data.get("seed"),
        )


class CandidateGenerator:
    """Procedurally identifies applicable rules and generates proposed candidate circuits."""

    def __init__(
        self,
        rules: Optional[Sequence[TransformationRule]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize generator with a library of transformation rules."""
        self.rules: List[TransformationRule] = (
            list(rules) if rules is not None else get_default_rules()
        )
        self.seed = seed

    def generate_candidates(
        self,
        circuit: CircuitData,
        seed: Optional[int] = None,
    ) -> List[CandidateCircuit]:
        """Find all applicable transformations and generate corresponding candidate circuits.

        Ensures parent circuit is not mutated and attaches traceable metadata.

        Args:
            circuit: The source circuit to transform.
            seed: Optional seed to record in candidate metadata.

        Returns:
            List of CandidateCircuit proposals.
        """
        active_seed = seed if seed is not None else self.seed
        parent_id = circuit.circuit_id
        candidates: List[CandidateCircuit] = []

        # Iterate over all registered rules deterministically
        for rule in self.rules:
            matches = rule.find_matches(circuit)
            for match in matches:
                # Apply transformation non-destructively
                transformed_circuit = rule.apply(circuit, match)

                # Compute preliminary cost
                prelim_cost = compute_circuit_cost(transformed_circuit)

                # Deterministic candidate ID
                loc_str = ",".join(str(idx) for idx in match.indices)
                candidate_signature = f"{parent_id}:{rule.name}:{loc_str}:{transformed_circuit.circuit_id}"
                cand_id = hashlib.sha256(candidate_signature.encode("utf-8")).hexdigest()[:16]

                candidate = CandidateCircuit(
                    candidate_id=cand_id,
                    parent_circuit_id=parent_id,
                    transformation_name=rule.name,
                    transformation_location=match.indices,
                    circuit=transformed_circuit,
                    preliminary_cost=prelim_cost,
                    verified=False,
                    verification_status="unverified",
                    seed=active_seed,
                )
                candidates.append(candidate)

        return candidates
