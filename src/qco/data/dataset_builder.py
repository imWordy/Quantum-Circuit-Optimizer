"""Ground-truth Dataset Builder for QCO.

Executes the non-ML ground truth pipeline:
Circuit -> Generate Candidates -> Verify Equivalence -> Compute Costs & Reductions -> Store Observations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from qco.circuits.representation import CircuitData
from qco.circuits.serialization import load_circuit, save_circuit
from qco.data.schemas import CandidateObservation
from qco.metrics.circuit_cost import compute_circuit_cost, compute_cost_reduction
from qco.transformations.candidate_generator import CandidateCircuit, CandidateGenerator
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.data.dataset_builder")


class DatasetBuilder:
    """Builds ground truth candidate observation datasets from raw quantum circuits."""

    def __init__(
        self,
        candidate_generator: Optional[CandidateGenerator] = None,
        tolerance: float = 1e-7,
    ) -> None:
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.tolerance = tolerance

    def process_circuit(
        self, circuit: CircuitData
    ) -> Tuple[List[CandidateObservation], List[CandidateCircuit]]:
        """Generate, verify, and measure all candidates for a single circuit.

        Args:
            circuit: The source circuit.

        Returns:
            Tuple of (observations, candidates).
        """
        candidates = self.candidate_generator.generate_candidates(circuit)
        parent_cost = compute_circuit_cost(circuit)
        observations: List[CandidateObservation] = []

        for cand in candidates:
            # Independent equivalence check (Phase 5)
            eq_res = check_equivalence(circuit, cand.circuit, tolerance=self.tolerance)

            # Cost and reduction measurement (Phase 4)
            cand_cost = compute_circuit_cost(cand.circuit)
            reduction = compute_cost_reduction(parent_cost, cand_cost)

            obs = CandidateObservation(
                parent_circuit_id=circuit.circuit_id,
                candidate_id=cand.candidate_id,
                transformation_type=cand.transformation_name,
                transformation_location=cand.transformation_location,
                parent_gate_count=parent_cost.total_gates,
                candidate_gate_count=cand_cost.total_gates,
                parent_cnot_count=parent_cost.cnot_count,
                candidate_cnot_count=cand_cost.cnot_count,
                parent_depth=parent_cost.depth,
                candidate_depth=cand_cost.depth,
                gate_reduction=reduction.gate_reduction,
                cnot_reduction=reduction.cnot_reduction,
                depth_reduction=reduction.depth_reduction,
                equivalent=eq_res.equivalent,
                matrix_distance=eq_res.matrix_distance if eq_res.matrix_distance is not None else -1.0,
                tolerance=self.tolerance,
            )
            observations.append(obs)

        return observations, candidates

    def build_dataset(
        self,
        circuits: Sequence[CircuitData],
        output_dir: Optional[Union[str, Path]] = None,
        save_circuit_definitions: bool = True,
    ) -> pd.DataFrame:
        """Process multiple circuits into a ground-truth observation dataset.

        Args:
            circuits: List of source circuits.
            output_dir: Optional directory to store records and JSON circuit definitions.
            save_circuit_definitions: If True, saves parent and candidate circuits to disk.

        Returns:
            Pandas DataFrame containing all observations.
        """
        all_observations: List[CandidateObservation] = []

        if output_dir:
            out_path = Path(output_dir)
            circuits_dir = out_path / "circuits"
            circuits_dir.mkdir(parents=True, exist_ok=True)
        else:
            circuits_dir = None

        for circuit in circuits:
            obs_list, cand_list = self.process_circuit(circuit)
            all_observations.extend(obs_list)

            if circuits_dir and save_circuit_definitions:
                save_circuit(circuit, circuits_dir / f"{circuit.circuit_id}.json")
                for cand in cand_list:
                    save_circuit(cand.circuit, circuits_dir / f"{cand.candidate_id}.json")

        df = pd.DataFrame([obs.to_dict() for obs in all_observations])

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path / "observations.csv", index=False)
            try:
                df.to_parquet(out_path / "observations.parquet", index=False)
            except Exception as e:
                logger.warning(f"Could not save parquet format ({e}); CSV saved successfully.")
            logger.info(f"Saved {len(df)} candidate observations to {out_path}")

        return df
