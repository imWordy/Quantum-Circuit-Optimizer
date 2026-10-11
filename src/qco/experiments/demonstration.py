"""End-to-End Final Demonstration for QCO — Phase 21.

Demonstrates the complete decision pipeline on an unseen quantum circuit:
Original Circuit
      |
      v
Candidate Generation
      |
      v
GNN Ranking
      |
      v
Exact Verification
      |
      v
Cost Evaluation
      |
      v
Next Iteration
      |
      v
Optimized Circuit

Produces formatted decision logs, step-by-step pipeline trace, and final comparison report:
Metric          Original    Optimized
--------------------------------------
Gate count          X           Y
CNOT count          X           Y
Depth               X           Y
Equivalent          --          YES
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from qco.circuits.representation import CircuitData
from qco.circuits.serialization import save_circuit
from qco.metrics.circuit_cost import CircuitCost, compute_circuit_cost
from qco.models.gnn import CircuitGNN
from qco.models.ranker import CandidateRanker, RankedCandidate
from qco.optimization.base import OptimizationResult, StepRecord
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer
from qco.transformations.candidate_generator import Candidate, CandidateGenerator
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.experiments.demonstration")


@dataclass
class IterationDecisionTrace:
    """Detailed record of a single iteration step through the entire decision pipeline."""

    iteration: int
    circuit_before_id: str
    num_candidates_generated: int
    ranked_candidates: List[Dict[str, Any]]  # Top candidate ranking scores and predicted benefits
    verified_candidate_id: str
    transformation_applied: str
    transformation_location: str
    exact_equivalence_verified: bool
    phase_difference: Optional[float]
    matrix_distance: float
    gate_count_before: int
    gate_count_after: int
    cnot_count_before: int
    cnot_count_after: int
    depth_before: int
    depth_after: int
    accepted: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DemonstrationReport:
    """Comprehensive final demonstration summary and report."""

    circuit_id: str
    num_qubits: int
    original_cost: CircuitCost
    optimized_cost: CircuitCost
    is_equivalent: bool
    iterations_run: int
    candidate_evaluations: int
    equivalence_checks: int
    runtime: float
    decision_traces: List[IterationDecisionTrace]
    final_circuit: CircuitData
    original_circuit: CircuitData

    def to_dict(self) -> Dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "num_qubits": self.num_qubits,
            "original_cost": asdict(self.original_cost),
            "optimized_cost": asdict(self.optimized_cost),
            "is_equivalent": self.is_equivalent,
            "iterations_run": self.iterations_run,
            "candidate_evaluations": self.candidate_evaluations,
            "equivalence_checks": self.equivalence_checks,
            "runtime": self.runtime,
            "decision_traces": [t.to_dict() for t in self.decision_traces],
            "original_circuit": self.original_circuit.to_dict(),
            "final_circuit": self.final_circuit.to_dict(),
        }

    def format_summary_table(self) -> str:
        """Render the required Phase 21 final report table."""
        equiv_str = "YES" if self.is_equivalent else "NO"
        lines = [
            "==================================================",
            "        QUANTUM CIRCUIT OPTIMIZATION REPORT       ",
            "==================================================",
            f"{'Metric':<16} {'Original':<12} {'Optimized':<12}",
            "--------------------------------------------------",
            f"{'Gate count':<16} {self.original_cost.total_gates:<12} {self.optimized_cost.total_gates:<12}",
            f"{'CNOT count':<16} {self.original_cost.cnot_count:<12} {self.optimized_cost.cnot_count:<12}",
            f"{'Depth':<16} {self.original_cost.depth:<12} {self.optimized_cost.depth:<12}",
            f"{'Equivalent':<16} {'--':<12} {equiv_str:<12}",
            "==================================================",
            f"Gate reduction : {self.original_cost.total_gates - self.optimized_cost.total_gates} gates",
            f"CNOT reduction : {self.original_cost.cnot_count - self.optimized_cost.cnot_count} CNOTs",
            f"Depth reduction: {self.original_cost.depth - self.optimized_cost.depth} layers",
            f"Search runtime : {self.runtime:.4f} s",
            "==================================================",
        ]
        return "\n".join(lines)


class FinalDemonstrator:
    """Runs and logs the full Phase 21 demonstration pipeline on unseen circuits."""

    def __init__(
        self,
        model: Optional[CircuitGNN] = None,
        max_iterations: int = 25,
        max_evaluations_per_iter: int = 15,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
    ) -> None:
        self.model = model or CircuitGNN(hidden_dim=64, gnn_type="gcn")
        self.max_iterations = max_iterations
        self.max_evaluations_per_iter = max_evaluations_per_iter
        self.tolerance = tolerance
        self.seed = seed
        self.ranker = CandidateRanker(model=self.model)

    def run_demonstration(
        self,
        unseen_circuit: CircuitData,
        verbose: bool = True,
    ) -> DemonstrationReport:
        """Run the end-to-end decision pipeline with full step logging.

        Stages traced per iteration:
        1. Original / Current Circuit state
        2. Procedural Candidate Generation
        3. GNN Ranking & Benefit Prediction
        4. Independent Exact Equivalence Verification
        5. Exact Cost Evaluation & Monotonic Acceptance
        """
        import time

        start_time = time.perf_counter()

        current = unseen_circuit
        original_cost = compute_circuit_cost(unseen_circuit)
        current_cost = original_cost

        decision_traces: List[IterationDecisionTrace] = []
        visited_ids = {current.circuit_id}

        candidate_evaluations = 0
        equivalence_checks = 0

        if verbose:
            logger.info("=== STARTING PHASE 21 FINAL DEMONSTRATION ===")
            logger.info(
                f"Input Unseen Circuit: ID={current.circuit_id}, Qubits={current.num_qubits}, "
                f"Gates={original_cost.total_gates}, CNOTs={original_cost.cnot_count}, Depth={original_cost.depth}"
            )

        for it in range(1, self.max_iterations + 1):
            # 1. Candidate Generation & GNN Ranking
            ranked: List[RankedCandidate] = self.ranker.rank_candidates(current)
            if not ranked:
                if verbose:
                    logger.info(f"Iteration {it}: No valid candidate transformations found. Converged.")
                break

            top_ranked_info = [
                {
                    "candidate_id": rc.candidate.candidate_id,
                    "rule": rc.candidate.transformation_name,
                    "location": rc.candidate.transformation_location,
                    "predicted_gate_reduction": float(rc.predicted_score),
                    "rank": rc.rank,
                }
                for rc in ranked[:5]
            ]

            accepted_in_iter = False

            # 2. Iterate through GNN-prioritized candidates
            for rc in ranked[: self.max_evaluations_per_iter]:
                cand = rc.candidate
                candidate_evaluations += 1

                if cand.circuit.circuit_id in visited_ids:
                    continue

                if cand.preliminary_cost.total_gates >= current_cost.total_gates:
                    continue

                # 3. Exact Verification Authority (Phase 5)
                equivalence_checks += 1
                equiv_res = check_equivalence(
                    current, cand.circuit, tolerance=self.tolerance
                )

                if not equiv_res.equivalent:
                    continue

                # 4. Cost Evaluation
                new_cost = compute_circuit_cost(cand.circuit)
                if new_cost.total_gates < current_cost.total_gates:
                    # Accepted
                    accepted_in_iter = True
                    visited_ids.add(cand.circuit.circuit_id)

                    trace = IterationDecisionTrace(
                        iteration=it,
                        circuit_before_id=current.circuit_id,
                        num_candidates_generated=len(ranked),
                        ranked_candidates=top_ranked_info,
                        verified_candidate_id=cand.candidate_id,
                        transformation_applied=cand.transformation_name,
                        transformation_location=cand.transformation_location,
                        exact_equivalence_verified=True,
                        phase_difference=equiv_res.phase_difference,
                        matrix_distance=float(equiv_res.matrix_distance if equiv_res.matrix_distance is not None else 0.0),
                        gate_count_before=current_cost.total_gates,
                        gate_count_after=new_cost.total_gates,
                        cnot_count_before=current_cost.cnot_count,
                        cnot_count_after=new_cost.cnot_count,
                        depth_before=current_cost.depth,
                        depth_after=new_cost.depth,
                        accepted=True,
                        reason=f"Beneficial verified reduction: {current_cost.total_gates} -> {new_cost.total_gates} gates",
                    )
                    decision_traces.append(trace)

                    if verbose:
                        logger.info(
                            f"Iteration {it} [ACCEPTED]: {cand.transformation_name} at {cand.transformation_location} | "
                            f"Gates: {current_cost.total_gates} -> {new_cost.total_gates} | "
                            f"CNOTs: {current_cost.cnot_count} -> {new_cost.cnot_count} | "
                            f"Equiv: YES (dist={equiv_res.matrix_distance:.2e})"
                        )

                    current = cand.circuit
                    current_cost = new_cost
                    break

            if not accepted_in_iter:
                if verbose:
                    logger.info(f"Iteration {it}: No candidate improved cost. Reached local minimum.")
                break

        runtime = time.perf_counter() - start_time

        # Final independent verification check between original input and final result
        final_equiv = check_equivalence(unseen_circuit, current, tolerance=self.tolerance)

        report = DemonstrationReport(
            circuit_id=unseen_circuit.circuit_id,
            num_qubits=unseen_circuit.num_qubits,
            original_cost=original_cost,
            optimized_cost=current_cost,
            is_equivalent=final_equiv.equivalent,
            iterations_run=len(decision_traces),
            candidate_evaluations=candidate_evaluations,
            equivalence_checks=equivalence_checks,
            runtime=runtime,
            decision_traces=decision_traces,
            final_circuit=current,
            original_circuit=unseen_circuit,
        )

        if verbose:
            print("\n" + report.format_summary_table())

        return report

    def save_demonstration(
        self,
        report: DemonstrationReport,
        output_dir: Union[str, Path] = "results/demonstration",
    ) -> None:
        """Export demonstration traces, circuits, and markdown report to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save JSON trace
        with open(out_path / "demonstration_report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save ASCII/Markdown report
        with open(out_path / "demonstration_report.txt", "w", encoding="utf-8") as f:
            f.write(report.format_summary_table())

        # Save circuit artifacts
        save_circuit(report.original_circuit, out_path / "original_circuit.json")
        save_circuit(report.final_circuit, out_path / "optimized_circuit.json")

        logger.info(f"Saved Phase 21 demonstration artifacts to {out_path}.")
