"""Schemas and data containers for verified candidate observations and datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CandidateObservation:
    """A verified candidate observation representing ground-truth optimization data."""

    parent_circuit_id: str
    candidate_id: str
    transformation_type: str
    transformation_location: Tuple[int, ...]

    parent_gate_count: int
    candidate_gate_count: int
    parent_cnot_count: int
    candidate_cnot_count: int
    parent_depth: int
    candidate_depth: int

    gate_reduction: int
    cnot_reduction: int
    depth_reduction: int

    equivalent: bool
    matrix_distance: float
    tolerance: float = 1e-7

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        d = asdict(self)
        d["transformation_location"] = list(self.transformation_location)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateObservation:
        """Construct observation from dictionary."""
        return cls(
            parent_circuit_id=data["parent_circuit_id"],
            candidate_id=data["candidate_id"],
            transformation_type=data["transformation_type"],
            transformation_location=tuple(data["transformation_location"]),
            parent_gate_count=data["parent_gate_count"],
            candidate_gate_count=data["candidate_gate_count"],
            parent_cnot_count=data["parent_cnot_count"],
            candidate_cnot_count=data["candidate_cnot_count"],
            parent_depth=data["parent_depth"],
            candidate_depth=data["candidate_depth"],
            gate_reduction=data["gate_reduction"],
            cnot_reduction=data["cnot_reduction"],
            depth_reduction=data["depth_reduction"],
            equivalent=data["equivalent"],
            matrix_distance=data["matrix_distance"],
            tolerance=data.get("tolerance", 1e-7),
        )
