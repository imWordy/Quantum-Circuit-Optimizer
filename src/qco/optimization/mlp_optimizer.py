"""Aggregate-Feature MLP Optimizer for QCO — Phase 18 / Phase 19 comparison."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Set

import torch

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.models.mlp_baseline import AggregateFeatureMLP, extract_aggregate_features
from qco.optimization.base import BaseOptimizer, OptimizationResult, StepRecord
from qco.transformations.candidate_generator import Candidate, CandidateGenerator
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.optimization.mlp_optimizer")


class MLPOptimizer(BaseOptimizer):
    """Optimizer that ranks candidates based on aggregate-feature MLP predictions."""

    def __init__(
        self,
        model: Optional[AggregateFeatureMLP] = None,
        max_iterations: int = 50,
        max_evaluations_per_iter: int = 20,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
        candidate_generator: Optional[CandidateGenerator] = None,
    ) -> None:
        super().__init__(max_iterations=max_iterations, tolerance=tolerance, seed=seed)
        self.model = model or AggregateFeatureMLP()
        self.model.eval()
        self.max_evaluations_per_iter = max_evaluations_per_iter
        self.candidate_generator = candidate_generator or CandidateGenerator()

    @property
    def name(self) -> str:
        return "MLPOptimizer"

    def optimize(self, circuit: CircuitData) -> OptimizationResult:
        """Run MLP-guided iterative optimization."""
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

            # Score candidates using MLP on candidate circuit's aggregate features
            scored_candidates = []
            for cand in candidates:
                feats = extract_aggregate_features(cand.circuit)
                with torch.no_grad():
                    pred_score = self.model(feats.unsqueeze(0)).item()
                # We prioritize candidates that reduce cost; predicted reduction or negative cost
                scored_candidates.append((cand, pred_score))

            # Sort descending by predicted score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            accepted_this_iter = False
            eval_budget = min(len(scored_candidates), self.max_evaluations_per_iter)

            for cand, score in scored_candidates[:eval_budget]:
                candidate_evaluations += 1

                if cand.circuit.circuit_id in visited_ids:
                    continue

                if cand.preliminary_cost.total_gates >= current_cost.total_gates:
                    continue

                equivalence_checks += 1
                equiv = check_equivalence(current, cand.circuit, tolerance=self.tolerance)

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
                                reason=f"Accepted MLP scored candidate: gates {current_cost.total_gates} -> {cand_cost.total_gates}",
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
