"""Dataset Splitting and Data Loading with Leakage Prevention for QCO — Phase 9.

Implements Grouped / Disjoint dataset splitting based on parent_circuit_id:
- Candidates derived from the same parent circuit are guaranteed to remain
  in the exact same split.
- Configurable train / validation / test ratios (default: 70% / 15% / 15%).
- Seed-controlled deterministic splitting.
- Disjointness validation preventing data leakage across splits.
- Serialization and loading of split datasets (observations and circuit references).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from qco.circuits.representation import CircuitData
from qco.circuits.serialization import load_circuit, save_circuit
from qco.data.schemas import ObservationRecord
from qco.utils.config import load_config
from qco.utils.logging import setup_logger

logger = setup_logger("qco.data.splits")


@dataclass
class DatasetSplits:
    """Container for split observation records and circuit stores."""

    train_records: List[ObservationRecord]
    val_records: List[ObservationRecord]
    test_records: List[ObservationRecord]
    circuit_store: Dict[str, CircuitData]

    @property
    def train_parent_ids(self) -> Set[str]:
        return {r.parent_circuit_id for r in self.train_records}

    @property
    def val_parent_ids(self) -> Set[str]:
        return {r.parent_circuit_id for r in self.val_records}

    @property
    def test_parent_ids(self) -> Set[str]:
        return {r.parent_circuit_id for r in self.test_records}

    def validate_leakage(self) -> bool:
        """Verify that parent circuit IDs are completely disjoint across all splits."""
        train_p = self.train_parent_ids
        val_p = self.val_parent_ids
        test_p = self.test_parent_ids

        train_val_overlap = train_p.intersection(val_p)
        train_test_overlap = train_p.intersection(test_p)
        val_test_overlap = val_p.intersection(test_p)

        if train_val_overlap:
            raise ValueError(f"Data leakage detected! Train and Val share parent IDs: {train_val_overlap}")
        if train_test_overlap:
            raise ValueError(f"Data leakage detected! Train and Test share parent IDs: {train_test_overlap}")
        if val_test_overlap:
            raise ValueError(f"Data leakage detected! Val and Test share parent IDs: {val_test_overlap}")

        return True

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save train, val, and test splits into target directory.

        Saves:
          - splits/train.jsonl, splits/val.jsonl, splits/test.jsonl
          - splits/train.csv, splits/val.csv, splits/test.csv
          - splits/split_metadata.json
          - circuits/: All referenced CircuitData JSON files
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        splits_map = {
            "train": self.train_records,
            "val": self.val_records,
            "test": self.test_records,
        }

        metadata = {
            "num_total_records": len(self.train_records) + len(self.val_records) + len(self.test_records),
            "num_total_parents": len(self.train_parent_ids | self.val_parent_ids | self.test_parent_ids),
            "splits": {},
        }

        for split_name, records in splits_map.items():
            jsonl_file = out_path / f"{split_name}.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r.to_dict()) + "\n")

            df = pd.DataFrame([r.to_dict() for r in records])
            df.to_csv(out_path / f"{split_name}.csv", index=False)

            parent_ids = {r.parent_circuit_id for r in records}
            metadata["splits"][split_name] = {
                "num_records": len(records),
                "num_parents": len(parent_ids),
            }

        # Save metadata
        with open(out_path / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save circuits
        circuits_dir = out_path / "circuits"
        circuits_dir.mkdir(exist_ok=True)
        for cid, circuit in self.circuit_store.items():
            save_circuit(circuit, circuits_dir / f"{cid}.json")

        logger.info(
            f"Saved dataset splits to {out_path}: "
            f"Train={len(self.train_records)}, Val={len(self.val_records)}, Test={len(self.test_records)}"
        )

    @classmethod
    def load(cls, split_dir: Union[str, Path]) -> DatasetSplits:
        """Load dataset splits from disk and validate leakage prevention.

        Args:
            split_dir: Directory containing train/val/test splits and circuits.

        Returns:
            DatasetSplits instance.
        """
        dir_path = Path(split_dir)
        splits_map: Dict[str, List[ObservationRecord]] = {}

        for name in ("train", "val", "test"):
            jsonl_file = dir_path / f"{name}.jsonl"
            if not jsonl_file.exists():
                raise FileNotFoundError(f"Missing split file: {jsonl_file}")

            records: List[ObservationRecord] = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(ObservationRecord.from_dict(json.loads(line)))
            splits_map[name] = records

        # Load circuits
        circuits_dir = dir_path / "circuits"
        circuit_store: Dict[str, CircuitData] = {}
        if circuits_dir.exists():
            for p in circuits_dir.glob("*.json"):
                circuit = load_circuit(p)
                circuit_store[circuit.circuit_id] = circuit

        splits = cls(
            train_records=splits_map["train"],
            val_records=splits_map["val"],
            test_records=splits_map["test"],
            circuit_store=circuit_store,
        )
        splits.validate_leakage()
        return splits


def split_dataset(
    records: Sequence[ObservationRecord],
    circuit_store: Optional[Dict[str, CircuitData]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = 42,
) -> DatasetSplits:
    """Split observation records by parent circuit ID to prevent data leakage.

    All candidates derived from the same parent circuit are placed strictly into
    the same split.

    Args:
        records: Collection of ObservationRecord instances.
        circuit_store: Optional dictionary of circuit_id -> CircuitData.
        train_ratio: Fraction of parent circuits allocated to training (default 0.70).
        val_ratio: Fraction allocated to validation (default 0.15).
        test_ratio: Fraction allocated to testing (default 0.15).
        seed: Random seed for reproducible splitting.

    Returns:
        DatasetSplits object containing disjoint train, val, and test records.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio})"
        )

    if not records:
        return DatasetSplits([], [], [], circuit_store or {})

    # 1. Group records by parent_circuit_id
    grouped_by_parent: Dict[str, List[ObservationRecord]] = {}
    for r in records:
        grouped_by_parent.setdefault(r.parent_circuit_id, []).append(r)

    unique_parents = sorted(grouped_by_parent.keys())
    num_parents = len(unique_parents)

    # 2. Shuffle parent circuit IDs deterministically
    rng = np.random.default_rng(seed)
    shuffled_parents = np.array(unique_parents)
    rng.shuffle(shuffled_parents)

    # 3. Calculate partition cutoffs
    n_train = int(np.floor(train_ratio * num_parents))
    n_val = int(np.floor(val_ratio * num_parents))
    # Remaining parents go to test
    train_parent_set = set(shuffled_parents[:n_train])
    val_parent_set = set(shuffled_parents[n_train : n_train + n_val])
    test_parent_set = set(shuffled_parents[n_train + n_val :])

    # In edge cases with very few circuits, ensure train has at least 1 if possible
    if num_parents >= 3 and len(train_parent_set) == 0:
        train_parent_set.add(shuffled_parents[0])
        val_parent_set.discard(shuffled_parents[0])
        test_parent_set.discard(shuffled_parents[0])

    train_records: List[ObservationRecord] = []
    val_records: List[ObservationRecord] = []
    test_records: List[ObservationRecord] = []

    for pid in train_parent_set:
        train_records.extend(grouped_by_parent[pid])
    for pid in val_parent_set:
        val_records.extend(grouped_by_parent[pid])
    for pid in test_parent_set:
        test_records.extend(grouped_by_parent[pid])

    splits = DatasetSplits(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        circuit_store=circuit_store or {},
    )

    # Validate leakage
    splits.validate_leakage()

    logger.info(
        f"Grouped split on {num_parents} parent circuits ({len(records)} records): "
        f"Train={len(train_parent_set)} parents ({len(train_records)} records), "
        f"Val={len(val_parent_set)} parents ({len(val_records)} records), "
        f"Test={len(test_parent_set)} parents ({len(test_records)} records)"
    )

    return splits


def split_dataset_from_config(
    records: Sequence[ObservationRecord],
    circuit_store: Optional[Dict[str, CircuitData]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DatasetSplits:
    """Split dataset using settings from central configuration."""
    if config is None:
        config = load_config()

    d_cfg = config.get("data", {})
    seed = config.get("seed", 42)

    return split_dataset(
        records=records,
        circuit_store=circuit_store,
        train_ratio=float(d_cfg.get("train_ratio", 0.70)),
        val_ratio=float(d_cfg.get("validation_ratio", 0.15)),
        test_ratio=float(d_cfg.get("test_ratio", 0.15)),
        seed=seed,
    )
