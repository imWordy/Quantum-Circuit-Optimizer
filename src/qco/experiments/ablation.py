"""Ablation Studies Framework for QCO — Phase 19.

Investigates which system components contribute to optimization performance:
1. Minimum Comparison (RQ2 & RQ4):
   - A: Random candidate selection (RandomOptimizer)
   - B: Greedy candidate selection (GreedyOptimizer)
   - C: Aggregate-feature ML (MLPOptimizer)
   - D: Graph-based GNN (GNNGuidedOptimizer)
   Testing whether structural graph information provides useful predictive guidance
   beyond aggregate circuit statistics.

2. Optional / Extended Ablations:
   - With vs. Without Feedback Learning (testing whether online feedback improves search)
   - Transformation Subsets (e.g. Clifford only vs full rotation rules)
   - Graph Edge Ablations (full dependency edges vs wire-only vs empty graph)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from qco.circuits.representation import CircuitData
from qco.experiments.benchmark import BenchmarkHarness, BenchmarkRecord
from qco.models.gnn import CircuitGNN
from qco.models.mlp_baseline import AggregateFeatureMLP
from qco.optimization.base import BaseOptimizer
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer
from qco.optimization.greedy_optimizer import GreedyOptimizer
from qco.optimization.mlp_optimizer import MLPOptimizer
from qco.optimization.random_optimizer import RandomOptimizer
from qco.transformations.candidate_generator import CandidateGenerator
from qco.transformations.rules import (
    CNOTCancellation,
    IdentityRemoval,
    RotationCombination,
    SelfInverseCancellation,
    TransformationRule,
    ZeroRotationRemoval,
    get_default_rules,
)
from qco.utils.logging import setup_logger

logger = setup_logger("qco.experiments.ablation")


@dataclass
class AblationStudyResult:
    """Standard container for ablation experiment outcomes."""

    study_name: str
    records: List[BenchmarkRecord]
    summary_df: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "records": [r.to_dict() for r in self.records],
            "summary": self.summary_df.to_dict(orient="records"),
        }


class AblationFramework:
    """Orchestrates controlled ablation experiments for QCO."""

    def __init__(
        self,
        test_circuits: Sequence[CircuitData],
        gnn_model: Optional[CircuitGNN] = None,
        mlp_model: Optional[AggregateFeatureMLP] = None,
        max_iterations: int = 20,
        max_evaluations_per_iter: int = 15,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
    ) -> None:
        self.test_circuits = list(test_circuits)
        self.gnn_model = gnn_model or CircuitGNN(hidden_dim=64, gnn_type="gcn")
        self.mlp_model = mlp_model or AggregateFeatureMLP(hidden_dim=32)
        self.max_iterations = max_iterations
        self.max_evaluations_per_iter = max_evaluations_per_iter
        self.tolerance = tolerance
        self.seed = seed

    def run_structural_vs_aggregate_ablation(
        self,
        custom_candidate_generator: Optional[CandidateGenerator] = None,
    ) -> AblationStudyResult:
        """Run the core Phase 19 minimum comparison: A, B, C, D.

        A: RandomOptimizer
        B: GreedyOptimizer
        C: MLPOptimizer (Aggregate statistics only)
        D: GNNGuidedOptimizer (Graph structure + node features)
        """
        logger.info("Running minimum ablation study: Random (A) vs Greedy (B) vs MLP (C) vs GNN (D)")

        cand_gen = custom_candidate_generator or CandidateGenerator()

        opt_random = RandomOptimizer(
            max_iterations=self.max_iterations,
            max_evaluations_per_iter=self.max_evaluations_per_iter,
            tolerance=self.tolerance,
            seed=self.seed,
            candidate_generator=cand_gen,
        )

        opt_greedy = GreedyOptimizer(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            seed=self.seed,
            candidate_generator=cand_gen,
        )

        opt_mlp = MLPOptimizer(
            model=self.mlp_model,
            max_iterations=self.max_iterations,
            max_evaluations_per_iter=self.max_evaluations_per_iter,
            tolerance=self.tolerance,
            seed=self.seed,
            candidate_generator=cand_gen,
        )

        opt_gnn = GNNGuidedOptimizer(
            model=self.gnn_model,
            max_iterations=self.max_iterations,
            max_evaluations_per_iter=self.max_evaluations_per_iter,
            tolerance=self.tolerance,
            seed=self.seed,
            candidate_generator=cand_gen,
        )

        harness = BenchmarkHarness(
            optimizers=[opt_random, opt_greedy, opt_mlp, opt_gnn]
        )
        harness.run_benchmark(self.test_circuits)
        summary = harness.compute_summary_table()

        return AblationStudyResult(
            study_name="structural_vs_aggregate",
            records=harness.records,
            summary_df=summary,
        )

    def run_rule_subset_ablation(
        self,
        rule_subsets: Optional[Dict[str, List[TransformationRule]]] = None,
    ) -> Dict[str, AblationStudyResult]:
        """Ablation on transformation rule subsets (e.g., Clifford cancellation vs Rotation combination).

        Compares GNNGuidedOptimizer performance across different subsets of transformation rules.
        """
        if rule_subsets is None:
            rule_subsets = {
                "cancellation_only": [
                    SelfInverseCancellation(),
                    CNOTCancellation(),
                    IdentityRemoval(),
                ],
                "rotations_only": [
                    RotationCombination(),
                    ZeroRotationRemoval(),
                    IdentityRemoval(),
                ],
                "all_rules": get_default_rules(),
            }

        results: Dict[str, AblationStudyResult] = {}
        for subset_name, rules in rule_subsets.items():
            logger.info(f"Running rule subset ablation for: {subset_name}")
            cand_gen = CandidateGenerator(rules=rules)
            opt_gnn = GNNGuidedOptimizer(
                model=self.gnn_model,
                max_iterations=self.max_iterations,
                max_evaluations_per_iter=self.max_evaluations_per_iter,
                tolerance=self.tolerance,
                seed=self.seed,
                candidate_generator=cand_gen,
                name=f"GNNGuidedOptimizer[{subset_name}]",
            )

            harness = BenchmarkHarness(optimizers=[opt_gnn])
            harness.run_benchmark(self.test_circuits)
            results[subset_name] = AblationStudyResult(
                study_name=f"rule_subset_{subset_name}",
                records=harness.records,
                summary_df=harness.compute_summary_table(),
            )

        return results

    def save_ablation_results(
        self,
        result: AblationStudyResult,
        output_dir: Union[str, Path],
    ) -> None:
        """Export ablation records and summary table to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        result.summary_df.to_csv(out_path / f"{result.study_name}_summary.csv", index=False)
        with open(out_path / f"{result.study_name}_data.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved ablation study '{result.study_name}' results to {out_path}.")
