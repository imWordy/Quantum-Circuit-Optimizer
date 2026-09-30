"""Greedy Baseline Optimizer for QCO — Phase 11.

At each iteration:
1. Generate all candidate transformations from current circuit.
2. Evaluate all candidates for validity and immediate improvement.
3. Verify candidates independently with Phase 5 equivalence checker.
4. Select the best immediate valid candidate:
   - Primary tie-breaker: maximum gate_reduction (total_gates reduction)
   - Secondary tie-breaker: maximum cnot_reduction
   - Tertiary tie-breaker: maximum depth_reduction
5. Stop when no strictly beneficial improvement exists, repeat cycle detected,
   or max iterations reached.
"""

from __future__ import annotations

import time
from typing import List, Optional, Set, Tuple

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.optimization.base import BaseOptimizer, OptimizationResult, StepRecord
from qco.transformations.candidate_generator import Candidate, CandidateGenerator
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.optimization.greedy")


class GreedyOptimizer(BaseOptimizer):
    """Baseline B: Deterministic greedy best-improvement candidate selection."""

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(max_iterations=max_iterations, tolerance=tolerance, seed=seed)
        self.candidate_generator = CandidateGenerator()

    @property
    def name(self) -> str:
        return "GreedyOptimizer"

    def optimize(self, circuit: CircuitData) -> OptimizationResult:
        """Execute greedy steepest-descent optimization on quantum circuit."""
        start_time = time.perf_counter()

        current = circuit
        initial_cost = compute_circuit_cost(circuit)
        current_cost = initial_cost

        visited_ids: Set[str] = {current.circuit_id}
        history: List[StepRecord] = []

        candidate_evaluations = 0
        equivalence_checks = 0
        iterations = 0
        converged = False

        for it in range(self.max_iterations):
            iterations += 1
            candidates = self.candidate_generator.generate_candidates(current)
            if not candidates:
                converged = True
                break

            best_candidate: Optional[Candidate] = None
            best_score: Tuple[int, int, int] = (0, 0, 0)  # (gate_red, cnot_red, depth_red)
            best_cand_cost = None

            for cand in candidates:
                candidate_evaluations += 1

                # Cycle detection
                if cand.circuit.circuit_id in visited_ids:
                    continue

                # Quick preliminary filter
                prelim_red = current_cost.total_gates - cand.preliminary_cost.total_gates
                if prelim_red <= 0:
                    continue

                # Verify exact equivalence
                equivalence_checks += 1
                equiv = check_equivalence(
                    current, cand.circuit, tolerance=self.tolerance
                )

                if equiv.equivalent:
                    cand_cost = compute_circuit_cost(cand.circuit)
                    gate_red = current_cost.total_gates - cand_cost.total_gates
                    cnot_red = current_cost.cnot_count - cand_cost.cnot_count
                    depth_red = current_cost.depth - cand_cost.depth

                    # Accept only strictly beneficial improvements
                    if gate_red > 0:
                        score = (gate_red, cnot_red, depth_red)
                        if best_candidate is None or score > best_score:
                            best_candidate = cand
                            best_score = score
                            best_cand_cost = cand_cost

            if best_candidate is not None and best_cand_cost is not None:
                history.append(
                    StepRecord(
                        iteration=it + 1,
                        candidate_id=best_candidate.candidate_id,
                        transformation_name=best_candidate.transformation_name,
                        transformation_location=best_candidate.transformation_location,
                        accepted=True,
                        gate_count_before=current_cost.total_gates,
                        gate_count_after=best_cand_cost.total_gates,
                        cnot_count_before=current_cost.cnot_count,
                        cnot_count_after=best_cand_cost.cnot_count,
                        depth_before=current_cost.depth,
                        depth_after=best_cand_cost.depth,
                        reason=f"Accepted best greedy candidate with reduction {best_score}",
                    )
                )
                current = best_candidate.circuit
                current_cost = best_cand_cost
                visited_ids.add(current.circuit_id)
            else:
                converged = True
                break

        runtime = time.perf_counter() - start_time

        # Final verification against original initial circuit
        final_equiv = check_equivalence(circuit, current, tolerance=self.tolerance)
        equivalence_checks += 1

        return OptimizationResult(
            optimizer_name=self.name,
            initial_circuit_id=circuit.circuit_id,
            final_circuit_id=current.circuit_id,
            initial_circuit=circuit,
            final_circuit=current,
            initial_cost=initial_cost,
            final_cost=current_cost,
            gate_reduction=initial_cost.total_gates - current_cost.total_gates,
            cnot_reduction=initial_cost.cnot_count - current_cost.cnot_count,
            depth_reduction=initial_cost.depth - current_cost.depth,
            candidate_evaluations=candidate_evaluations,
            equivalence_checks=equivalence_checks,
            iterations=iterations,
            runtime=runtime,
            converged=converged,
            is_equivalent=final_equiv.equivalent,
            history=history,
        )
