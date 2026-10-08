"""Random Search Baseline Optimizer for QCO — Phase 11.

At each iteration:
1. Generate valid candidate transformations from current circuit.
2. Select candidate at random.
3. Verify exact equivalence using Phase 5 equivalence checker.
4. Accept only if it improves the primary cost (gate count reduction > 0).
5. Stop when no beneficial candidate is found, max iterations reached, or repeat cycle detected.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Set

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.optimization.base import BaseOptimizer, OptimizationResult, StepRecord
from qco.transformations.candidate_generator import CandidateGenerator
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.optimization.random")


class RandomOptimizer(BaseOptimizer):
    """Baseline A: Iterative random candidate selection."""

    def __init__(
        self,
        max_iterations: int = 50,
        max_evaluations_per_iter: int = 10,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
        candidate_generator: Optional[CandidateGenerator] = None,
    ) -> None:
        super().__init__(max_iterations=max_iterations, tolerance=tolerance, seed=seed)
        self.max_evaluations_per_iter = max_evaluations_per_iter
        self.candidate_generator = candidate_generator or CandidateGenerator()

    @property
    def name(self) -> str:
        return "RandomOptimizer"

    def optimize(self, circuit: CircuitData) -> OptimizationResult:
        """Execute iterative random search on quantum circuit."""
        start_time = time.perf_counter()
        rng = random.Random(self.seed)

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

            # Shuffle candidates to explore randomly
            shuffled = list(candidates)
            rng.shuffle(shuffled)

            accepted_this_iter = False
            # Check up to max_evaluations_per_iter random candidates
            eval_budget = min(len(shuffled), self.max_evaluations_per_iter)

            for cand in shuffled[:eval_budget]:
                candidate_evaluations += 1

                # Cycle detection
                if cand.circuit.circuit_id in visited_ids:
                    continue

                # Primary cost check (preliminary gate reduction)
                if cand.preliminary_cost.total_gates >= current_cost.total_gates:
                    continue

                # Verify equivalence independently (Phase 5)
                equivalence_checks += 1
                equiv = check_equivalence(
                    current, cand.circuit, tolerance=self.tolerance
                )

                if equiv.equivalent:
                    cand_cost = compute_circuit_cost(cand.circuit)
                    if cand_cost.total_gates < current_cost.total_gates:
                        history.append(
                            StepRecord(
                                iteration=it + 1,
                                candidate_id=cand.candidate_id,
                                transformation_name=cand.transformation_name,
                                transformation_location=cand.transformation_location,
                                accepted=True,
                                gate_count_before=current_cost.total_gates,
                                gate_count_after=cand_cost.total_gates,
                                cnot_count_before=current_cost.cnot_count,
                                cnot_count_after=cand_cost.cnot_count,
                                depth_before=current_cost.depth,
                                depth_after=cand_cost.depth,
                                reason="Accepted beneficial random candidate",
                            )
                        )
                        current = cand.circuit
                        current_cost = cand_cost
                        visited_ids.add(current.circuit_id)
                        accepted_this_iter = True
                        break

            if not accepted_this_iter:
                converged = True
                break

        runtime = time.perf_counter() - start_time

        # Final independent verification against original circuit
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
