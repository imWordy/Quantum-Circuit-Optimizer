"""Controlled Experimental Benchmark Framework for QCO — Phase 18.

Runs a rigorously fair, controlled comparative experiment across:
1. Random Optimizer (Baseline A)
2. Greedy Optimizer (Baseline B)
3. GNN-Guided Optimizer (Core Method)
4. Aggregate-Feature MLP Optimizer (Baseline C)

Guarantees of Fair Comparison:
- Identical held-out test circuit set
- Identical transformation rules (default rules library)
- Identical search budget (max iterations and candidate evaluation limits)
- Identical equivalence verification authority (Phase 5 unitary tolerance)
- Identical termination conditions

Collects all required benchmark fields:
- circuit_id, method
- initial_gate_count, final_gate_count
- initial_cnot_count, final_cnot_count
- initial_depth, final_depth
- gate_reduction, cnot_reduction, depth_reduction
- candidate_evaluations, equivalence_checks
- iterations, runtime
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from qco.circuits.representation import CircuitData
from qco.optimization.base import BaseOptimizer, OptimizationResult
from qco.utils.logging import setup_logger

logger = setup_logger("qco.experiments.benchmark")


@dataclass
class BenchmarkRecord:
    """Standardized record for one optimizer run on a test circuit."""

    circuit_id: str
    method: str
    initial_gate_count: int
    final_gate_count: int
    initial_cnot_count: int
    final_cnot_count: int
    initial_depth: int
    final_depth: int
    gate_reduction: int
    cnot_reduction: int
    depth_reduction: int
    candidate_evaluations: int
    equivalence_checks: int
    iterations: int
    runtime: float
    is_equivalent: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_result(cls, res: OptimizationResult) -> BenchmarkRecord:
        return cls(
            circuit_id=res.initial_circuit_id,
            method=res.optimizer_name,
            initial_gate_count=res.initial_cost.total_gates,
            final_gate_count=res.final_cost.total_gates,
            initial_cnot_count=res.initial_cost.cnot_count,
            final_cnot_count=res.final_cost.cnot_count,
            initial_depth=res.initial_cost.depth,
            final_depth=res.final_cost.depth,
            gate_reduction=res.gate_reduction,
            cnot_reduction=res.cnot_reduction,
            depth_reduction=res.depth_reduction,
            candidate_evaluations=res.candidate_evaluations,
            equivalence_checks=res.equivalence_checks,
            iterations=res.iterations,
            runtime=res.runtime,
            is_equivalent=res.is_equivalent,
        )


class BenchmarkHarness:
    """Manages controlled comparative experiments across multiple optimizers on a test suite."""

    def __init__(self, optimizers: Sequence[BaseOptimizer]) -> None:
        self.optimizers = list(optimizers)
        self.records: List[BenchmarkRecord] = []

    def run_benchmark(self, test_circuits: Sequence[CircuitData]) -> pd.DataFrame:
        """Run all configured optimizers on every test circuit.

        Args:
            test_circuits: Sequence of held-out test CircuitData.

        Returns:
            pandas DataFrame containing all benchmark observation rows.
        """
        logger.info(
            f"Starting comparative benchmark: {len(self.optimizers)} methods across {len(test_circuits)} test circuits."
        )

        for c_idx, circuit in enumerate(test_circuits, start=1):
            for opt in self.optimizers:
                res = opt.optimize(circuit)
                record = BenchmarkRecord.from_result(res)
                self.records.append(record)

            if c_idx % 5 == 0 or c_idx == len(test_circuits):
                logger.info(f"Processed {c_idx}/{len(test_circuits)} test circuits.")

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records])

    def compute_summary_table(self) -> pd.DataFrame:
        """Aggregate mean metrics grouped by method."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        summary = (
            df.groupby("method")
            .agg(
                num_circuits=("circuit_id", "count"),
                avg_gate_reduction=("gate_reduction", "mean"),
                avg_cnot_reduction=("cnot_reduction", "mean"),
                avg_depth_reduction=("depth_reduction", "mean"),
                avg_evaluations=("candidate_evaluations", "mean"),
                avg_checks=("equivalence_checks", "mean"),
                avg_runtime_sec=("runtime", "mean"),
                all_equivalent=("is_equivalent", "all"),
            )
            .reset_index()
        )
        return summary

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save raw and summary results to CSV and JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(out_path / "benchmark_raw.csv", index=False)

        summary_df = self.compute_summary_table()
        summary_df.to_csv(out_path / "benchmark_summary.csv", index=False)

        with open(out_path / "benchmark_raw.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)

        logger.info(f"Saved benchmark results to {out_path}.")
