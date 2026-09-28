"""Procedural Candidate Generator for QCO — Phase 7.

Generates all possible transformed circuit candidates from a current circuit
by systematically applying each valid transformation rule at every applicable
location. Does not rank or select candidates — that is the GNN's role.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import CircuitCost, compute_circuit_cost
from qco.transformations.rules import (
    TransformationMatch,
    TransformationRule,
    get_default_rules,
)
from qco.utils.logging import setup_logger

logger = setup_logger("qco.transformations.candidate_generator")


@dataclass
class Candidate:
    """A proposed transformed circuit — only a proposal until independently verified."""

    candidate_id: str
    parent_circuit_id: str
    transformation_name: str
    transformation_location: str      # Human-readable description of where rule applied
    transformation_indices: List[int]  # Gate indices in parent that participated
    circuit: CircuitData
    preliminary_cost: CircuitCost
    verification_status: str = "unverified"   # "unverified" | "verified_equivalent" | "verified_nonequivalent"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate metadata (excluding full circuit, which is stored by reference)."""
        return {
            "candidate_id": self.candidate_id,
            "parent_circuit_id": self.parent_circuit_id,
            "transformation_name": self.transformation_name,
            "transformation_location": self.transformation_location,
            "transformation_indices": self.transformation_indices,
            "circuit_id": self.circuit.circuit_id,
            "preliminary_gate_count": self.preliminary_cost.total_gates,
            "preliminary_cnot_count": self.preliminary_cost.cnot_count,
            "preliminary_depth": self.preliminary_cost.depth,
            "verification_status": self.verification_status,
        }


class CandidateGenerator:
    """Generates all candidate transformed circuits from a given circuit.

    Uses all provided transformation rules to enumerate every applicable
    transformation at every valid location. Does NOT select or rank candidates.
    """

    def __init__(
        self,
        rules: Optional[Sequence[TransformationRule]] = None,
    ) -> None:
        """Initialize generator with an ordered list of transformation rules.

        Args:
            rules: List of TransformationRule instances. Defaults to get_default_rules().
        """
        self.rules = list(rules) if rules is not None else get_default_rules()

    def generate_candidates(self, circuit: CircuitData) -> List[Candidate]:
        """Generate all candidates from a circuit by applying every rule at every match.

        Args:
            circuit: Source CircuitData to transform. Must not be mutated.

        Returns:
            List of Candidate objects, each representing one possible transformation.
        """
        candidates: List[Candidate] = []

        for rule in self.rules:
            matches = rule.find_matches(circuit)

            for match in matches:
                try:
                    transformed = rule.apply(circuit, match)
                except Exception as e:
                    logger.warning(
                        f"Rule '{rule.name}' failed to apply at {match.indices}: {e}"
                    )
                    continue

                preliminary_cost = compute_circuit_cost(transformed)

                candidate = Candidate(
                    candidate_id=str(uuid.uuid4()),
                    parent_circuit_id=circuit.circuit_id,
                    transformation_name=rule.name,
                    transformation_location=match.description,
                    transformation_indices=list(match.indices),
                    circuit=transformed,
                    preliminary_cost=preliminary_cost,
                    verification_status="unverified",
                )
                candidates.append(candidate)

        logger.debug(
            f"Circuit {circuit.circuit_id}: generated {len(candidates)} candidates "
            f"across {len(self.rules)} rules."
        )
        return candidates

    def generate_candidates_from_batch(
        self, circuits: Sequence[CircuitData]
    ) -> Dict[str, List[Candidate]]:
        """Generate candidates for each circuit in a batch.

        Args:
            circuits: Sequence of CircuitData instances.

        Returns:
            Dict mapping each circuit_id to its list of Candidate objects.
        """
        result: Dict[str, List[Candidate]] = {}
        for circuit in circuits:
            result[circuit.circuit_id] = self.generate_candidates(circuit)
        return result
