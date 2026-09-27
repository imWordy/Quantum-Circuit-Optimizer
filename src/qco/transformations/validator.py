"""Transformation Rule Validator for QCO.

Validates that proposed transformations:
1. Preserve structural circuit constraints.
2. Produce mathematically equivalent circuits using the Phase 5 equivalence checker.
3. Deliver the expected reduction effect without unforeseen side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import compute_cost_reduction
from qco.transformations.rules import TransformationMatch, TransformationRule
from qco.verification.equivalence import EquivalenceResult, check_equivalence


@dataclass(frozen=True)
class RuleValidationResult:
    """Report on whether applying a transformation rule satisfies correctness."""

    valid: bool
    rule_name: str
    match: TransformationMatch
    equivalence_result: EquivalenceResult
    actual_gate_reduction: int
    actual_cnot_reduction: int
    error_message: Optional[str] = None


class RuleValidator:
    """Validates transformation rule applications against independent verification."""

    def __init__(self, tolerance: float = 1e-7) -> None:
        self.tolerance = tolerance

    def validate_transformation(
        self,
        rule: TransformationRule,
        circuit: CircuitData,
        match: TransformationMatch,
    ) -> RuleValidationResult:
        """Apply rule at match location and independently verify validity and equivalence.

        Args:
            rule: TransformationRule to apply.
            circuit: Original CircuitData.
            match: TransformationMatch location.

        Returns:
            RuleValidationResult.
        """
        try:
            transformed = rule.apply(circuit, match)
        except Exception as e:
            return RuleValidationResult(
                valid=False,
                rule_name=rule.name,
                match=match,
                equivalence_result=EquivalenceResult(
                    equivalent=False,
                    tolerance=self.tolerance,
                    comparison_method="none",
                    global_phase_handling="none",
                    reason=f"Application crashed: {str(e)}",
                ),
                actual_gate_reduction=0,
                actual_cnot_reduction=0,
                error_message=f"Failed to apply rule: {str(e)}",
            )

        # Independent equivalence check (Phase 5)
        equiv_res = check_equivalence(circuit, transformed, tolerance=self.tolerance)

        # Cost reduction calculation (Phase 4)
        red = compute_cost_reduction(circuit, transformed)

        is_valid = bool(equiv_res.equivalent and transformed.num_qubits == circuit.num_qubits)
        error_msg = None if is_valid else f"Transformation violated equivalence: {equiv_res.reason}"

        return RuleValidationResult(
            valid=is_valid,
            rule_name=rule.name,
            match=match,
            equivalence_result=equiv_res,
            actual_gate_reduction=red.gate_reduction,
            actual_cnot_reduction=red.cnot_reduction,
            error_message=error_msg,
        )
