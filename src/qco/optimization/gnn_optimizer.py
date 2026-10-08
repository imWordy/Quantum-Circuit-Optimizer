"""GNN-Guided Iterative Quantum Circuit Optimizer — Phase 16.

Integrates the deterministic verification pipeline with the trained Graph Neural Network:

Core loop:
    current = original circuit
    while stopping condition not met:
        candidates = generate_candidates(current)
        ranked_candidates = GNN.rank(candidates)
        for ranked_cand in ranked_candidates:
            # Cycle avoidance check
            if ranked_cand.candidate.circuit.circuit_id in visited_ids:
                continue
            # Phase 5 exact quantum equivalence verification
            equiv = verify_equivalence(current, ranked_cand.candidate.circuit)
            if equiv.equivalent:
                # Phase 4 exact cost evaluation
                cand_cost = compute_circuit_cost(ranked_cand.candidate.circuit)
                # Primary acceptance rule: strictly beneficial improvement in gate count
                if cand_cost.total_gates < current_cost.total_gates:
                    current = ranked_cand.candidate.circuit
                    break
        if no beneficial candidate accepted:
            stop

Guarantees:
- Every accepted candidate is independently verified equivalent (Phase 5).
- Primary cost (total gate count) strictly decreases; never increases.
- Deterministic cycle prevention avoids infinite loops.
- Configurable maximum iterations and evaluation budgets.
- Full trace of intermediate steps and metrics recorded.
"""

from __future__ import annotations

import time
from typing import List, Optional, Set

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.models.gnn import CircuitGNN
from qco.models.ranker import CandidateRanker, RankedCandidate
from qco.optimization.base import BaseOptimizer, OptimizationResult, StepRecord
from qco.transformations.candidate_generator import CandidateGenerator
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.optimization.gnn_optimizer")


class GNNGuidedOptimizer(BaseOptimizer):
    """GNN-Guided Iterative Optimizer integrating procedural rewriting, GNN ranking, and verification."""

    def __init__(
        self,
        model: CircuitGNN,
        max_iterations: int = 50,
        max_evaluations_per_iter: int = 20,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
        candidate_generator: Optional[CandidateGenerator] = None,
        feedback_collector: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize GNN-guided optimizer.

        Args:
            model: Trained CircuitGNN model used to prioritize candidate evaluations.
            max_iterations: Maximum outer optimization loop iterations.
            max_evaluations_per_iter: Maximum candidates to verify per iteration.
            tolerance: Numerical tolerance for quantum equivalence verification.
            seed: Seed for reproducibility if needed.
            candidate_generator: Generator producing candidate transformations.
            feedback_collector: Optional collector instance (e.g. FeedbackLoopManager).
            name: Optional custom name identifier.
        """
        super().__init__(max_iterations=max_iterations, tolerance=tolerance, seed=seed)
        self.model = model
        self.ranker = CandidateRanker(model=model, candidate_generator=candidate_generator)
        self.max_evaluations_per_iter = max_evaluations_per_iter
        self.feedback_collector = feedback_collector
        self._name = name or "GNNGuidedOptimizer"

    @property
    def name(self) -> str:
        return self._name

    def optimize(self, circuit: CircuitData) -> OptimizationResult:
        """Run GNN-guided iterative optimization on a quantum circuit.

        Args:
            circuit: Starting CircuitData.

        Returns:
            OptimizationResult containing final circuit, cost reductions, and trace.
        """
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

            # 1. Generate, score, and rank candidates using the GNN
            ranked_candidates: List[RankedCandidate] = self.ranker.rank_candidates(current)
            if not ranked_candidates:
                converged = True
                break

            accepted_this_iter = False
            eval_budget = min(len(ranked_candidates), self.max_evaluations_per_iter)

            # 2. Evaluate highest-priority candidates first
            for rc in ranked_candidates[:eval_budget]:
                candidate_evaluations += 1
                cand = rc.candidate

                # Cycle detection
                if cand.circuit.circuit_id in visited_ids:
                    continue

                # Quick preliminary check
                if cand.preliminary_cost.total_gates >= current_cost.total_gates:
                    continue

                # 3. Independent equivalence verification (Phase 5 authority)
                equivalence_checks += 1
                equiv = check_equivalence(
                    current, cand.circuit, tolerance=self.tolerance
                )

                cand_cost = compute_circuit_cost(cand.circuit)
                actual_benefit = current_cost.total_gates - cand_cost.total_gates

                # Record feedback if collector is attached
                if self.feedback_collector is not None:
                    from qco.models.feedback import FeedbackRecord
                    fb_rec = FeedbackRecord(
                        model_version=getattr(self.feedback_collector, "current_model_version", "v1"),
                        circuit_id=current.circuit_id,
                        candidate_id=cand.candidate_id,
                        transformation_type=cand.transformation_name,
                        transformation_location=cand.transformation_location,
                        predicted_benefit=rc.predicted_score,
                        actual_benefit=actual_benefit,
                        is_equivalent=equiv.equivalent,
                        candidate_gate_count=cand_cost.total_gates,
                        candidate_cnot_count=cand_cost.cnot_count,
                        candidate_depth=cand_cost.depth,
                        optimizer_iteration=it + 1,
                    )
                    self.feedback_collector.record_feedback(fb_rec)

                if equiv.equivalent:
                    # Primary acceptance rule: strictly beneficial improvement
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
                                reason=(
                                    f"Accepted GNN rank #{rc.rank} (score {rc.predicted_score:.3f}): "
                                    f"gates {current_cost.total_gates} -> {cand_cost.total_gates}"
                                ),
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

        # Final end-to-end equivalence verification against original initial circuit
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
