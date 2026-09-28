"""Ground-Truth Dataset Builder for QCO — Phase 8.

Implements the full non-ML pipeline:
  Initial circuit
    -> Generate candidates (Phase 7)
    -> Verify candidates (Phase 5)
    -> Calculate costs (Phase 4)
    -> Calculate actual improvements
    -> Store observation

Ground-truth principle: training targets come ONLY from actual verified outcomes.
GNN predictions must never be used as labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from qco.circuits.representation import CircuitData
from qco.circuits.serialization import save_circuit
from qco.data.schemas import ObservationRecord
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.transformations.candidate_generator import Candidate, CandidateGenerator
from qco.transformations.rules import get_default_rules, TransformationRule
from qco.utils.logging import setup_logger
from qco.verification.equivalence import check_equivalence

logger = setup_logger("qco.data.dataset_builder")


class DatasetBuilder:
    """Builds a verified ground-truth dataset from a collection of circuits.

    Pipeline per circuit:
    1. Generate candidates via CandidateGenerator (Phase 7)
    2. Independently verify each candidate via check_equivalence (Phase 5)
    3. Compute cost metrics on parent and candidate (Phase 4)
    4. Compute actual improvements
    5. Store observation record
    """

    def __init__(
        self,
        rules: Optional[Sequence[TransformationRule]] = None,
        equivalence_tolerance: float = 1e-7,
    ) -> None:
        self.generator = CandidateGenerator(
            rules=list(rules) if rules is not None else get_default_rules()
        )
        self.equivalence_tolerance = equivalence_tolerance
        self._records: List[ObservationRecord] = []
        self._circuit_store: Dict[str, CircuitData] = {}  # id -> CircuitData reference

    @property
    def records(self) -> List[ObservationRecord]:
        """Return all stored observation records."""
        return list(self._records)

    @property
    def circuit_store(self) -> Dict[str, CircuitData]:
        """Return mapping of circuit_id -> CircuitData for reconstruction."""
        return dict(self._circuit_store)

    def process_circuit(self, circuit: CircuitData) -> List[ObservationRecord]:
        """Run the full pipeline for one circuit and accumulate records.

        Args:
            circuit: Parent CircuitData to process.

        Returns:
            List of ObservationRecord for candidates generated from this circuit.
        """
        # Store parent circuit reference
        if circuit.circuit_id not in self._circuit_store:
            self._circuit_store[circuit.circuit_id] = circuit

        parent_cost = compute_circuit_cost(circuit)
        candidates = self.generator.generate_candidates(circuit)

        new_records: List[ObservationRecord] = []
        for candidate in candidates:
            # Independent equivalence verification (Phase 5)
            equiv_result = check_equivalence(
                circuit,
                candidate.circuit,
                tolerance=self.equivalence_tolerance,
            )
            equivalent = equiv_result.equivalent

            # Compute verified candidate cost (Phase 4)
            candidate_cost = compute_circuit_cost(candidate.circuit)

            # Actual improvements (ground truth — never from GNN)
            gate_reduction = parent_cost.total_gates - candidate_cost.total_gates
            cnot_reduction = parent_cost.cnot_count - candidate_cost.cnot_count
            depth_reduction = parent_cost.depth - candidate_cost.depth

            # Store candidate circuit reference for reconstruction
            if candidate.circuit.circuit_id not in self._circuit_store:
                self._circuit_store[candidate.circuit.circuit_id] = candidate.circuit

            record = ObservationRecord(
                parent_circuit_id=circuit.circuit_id,
                candidate_id=candidate.candidate_id,
                transformation_type=candidate.transformation_name,
                transformation_location=candidate.transformation_location,
                parent_gate_count=parent_cost.total_gates,
                parent_cnot_count=parent_cost.cnot_count,
                parent_depth=parent_cost.depth,
                candidate_gate_count=candidate_cost.total_gates,
                candidate_cnot_count=candidate_cost.cnot_count,
                candidate_depth=candidate_cost.depth,
                gate_reduction=gate_reduction,
                cnot_reduction=cnot_reduction,
                depth_reduction=depth_reduction,
                equivalent=equivalent,
                candidate_circuit_id=candidate.circuit.circuit_id,
            )
            self._records.append(record)
            new_records.append(record)

        logger.debug(
            f"Circuit {circuit.circuit_id}: "
            f"{len(candidates)} candidates -> {sum(1 for r in new_records if r.equivalent)} equivalent."
        )
        return new_records

    def process_batch(self, circuits: Sequence[CircuitData]) -> None:
        """Process a batch of circuits, accumulating all observations.

        Args:
            circuits: Sequence of CircuitData to process.
        """
        for i, circuit in enumerate(circuits):
            self.process_circuit(circuit)
            if (i + 1) % 50 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(circuits)} circuits, "
                    f"{len(self._records)} total records."
                )

        logger.info(
            f"Dataset built: {len(circuits)} circuits, {len(self._records)} total observations."
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self._records])

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save dataset records to disk.

        Saves:
          - observations.jsonl: one JSON object per line (one record per candidate)
          - observations.csv: tabular version
          - circuits/: JSON files for each unique circuit

        Args:
            output_dir: Directory to write dataset to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSONL
        jsonl_path = output_dir / "observations.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict()) + "\n")

        # Save CSV
        df = self.to_dataframe()
        df.to_csv(output_dir / "observations.csv", index=False)

        # Save circuit references
        circuits_dir = output_dir / "circuits"
        circuits_dir.mkdir(exist_ok=True)
        for cid, circuit in self._circuit_store.items():
            save_circuit(circuit, circuits_dir / f"{cid}.json")

        logger.info(
            f"Dataset saved to {output_dir}: "
            f"{len(self._records)} records, {len(self._circuit_store)} circuits."
        )

    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> Tuple[List[ObservationRecord], Dict[str, CircuitData]]:
        """Load a saved dataset from disk.

        Args:
            output_dir: Directory where dataset was saved.

        Returns:
            Tuple of (records, circuit_store).
        """
        from qco.circuits.serialization import load_circuit

        output_dir = Path(output_dir)
        jsonl_path = output_dir / "observations.jsonl"
        circuits_dir = output_dir / "circuits"

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ObservationRecord.from_dict(json.loads(line)))

        circuit_store: Dict[str, CircuitData] = {}
        if circuits_dir.exists():
            for cpath in circuits_dir.glob("*.json"):
                circuit = load_circuit(cpath)
                circuit_store[circuit.circuit_id] = circuit

        logger.info(f"Loaded {len(records)} records, {len(circuit_store)} circuits from {output_dir}.")
        return records, circuit_store
