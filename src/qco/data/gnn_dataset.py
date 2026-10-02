"""PyTorch Geometric Dataset Interface for GNN Candidate Ranking — Phase 12.

Prepares graph-learning data pairs:
  Each example provides:
    1. Circuit Graph (torch_geometric.data.Data) with x, edge_index, edge_attr
    2. Candidate transformation feature vector (candidate_features)
    3. Target regression value (y: gate_reduction)

Supports:
- Construction from ObservationRecord and circuit_store
- PyG DataLoader batching via custom/standard collation
- Zero test-label leakage into input features
- In-memory dataset handling and serialization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from qco.circuits.representation import CircuitData
from qco.data.candidate_features import CANDIDATE_FEAT_DIM, extract_candidate_features
from qco.data.dataset import DatasetSplits
from qco.data.schemas import ObservationRecord
from qco.graphs.circuit_graph import circuit_to_graph
from qco.utils.logging import setup_logger

logger = setup_logger("qco.data.gnn_dataset")


class CircuitCandidateData(Data):
    """PyTorch Geometric Data subclass representing a circuit + candidate proposal.

    Attributes:
        x: Node feature tensor [num_nodes, node_feat_dim]
        edge_index: Graph connectivity tensor [2, num_edges]
        edge_attr: Edge feature tensor [num_edges, edge_feat_dim]
        candidate_features: 1D tensor [CANDIDATE_FEAT_DIM] describing the candidate
        y: Regression target [1] (actual gate_reduction)
        parent_circuit_id: String ID of parent
        candidate_id: String ID of candidate
    """

    def __inc__(self, key: str, value: Any, *args, **kwargs):
        # Default increment for node indexing
        if key == "edge_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class GNNDataset(Dataset):
    """Dataset providing (circuit_graph, candidate_features, target) samples for GNN training/ranking."""

    def __init__(
        self,
        records: Sequence[ObservationRecord],
        circuit_store: Dict[str, CircuitData],
        target_field: str = "gate_reduction",
    ) -> None:
        """Initialize GNN dataset.

        Args:
            records: List of verified ObservationRecord instances.
            circuit_store: Mapping of circuit_id -> CircuitData.
            target_field: Field name to use as regression target (default: 'gate_reduction').
        """
        self.records = list(records)
        self.circuit_store = circuit_store
        self.target_field = target_field
        self._data_list: List[CircuitCandidateData] = []
        self._build_data_objects()

    def _build_data_objects(self) -> None:
        """Pre-construct graph Data objects with candidate features and regression targets."""
        for rec in self.records:
            parent_id = rec.parent_circuit_id
            if parent_id not in self.circuit_store:
                logger.warning(f"Parent circuit {parent_id} not found in circuit_store; skipping.")
                continue

            parent_circuit = self.circuit_store[parent_id]

            # 1. Circuit Graph representation (Phase 10)
            graph_data = circuit_to_graph(parent_circuit)

            # 2. Candidate transformation features (Phase 12)
            cand_feats = extract_candidate_features(rec, parent_circuit=parent_circuit)

            # 3. Target regression value (e.g. gate_reduction)
            target_val = float(getattr(rec, self.target_field, 0.0))
            y = torch.tensor([target_val], dtype=torch.float32)

            item = CircuitCandidateData(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr,
                candidate_features=cand_feats.unsqueeze(0),  # [1, CANDIDATE_FEAT_DIM]
                y=y,
                num_nodes=graph_data.num_nodes,
                parent_circuit_id=parent_id,
                candidate_id=rec.candidate_id,
            )
            self._data_list.append(item)

        logger.debug(f"Constructed GNNDataset with {len(self._data_list)} examples.")

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> CircuitCandidateData:
        return self._data_list[idx]

    @classmethod
    def from_splits(cls, splits: DatasetSplits, target_field: str = "gate_reduction") -> Dict[str, GNNDataset]:
        """Create train, val, and test GNNDatasets from a DatasetSplits instance."""
        return {
            "train": cls(splits.train_records, splits.circuit_store, target_field=target_field),
            "val": cls(splits.val_records, splits.circuit_store, target_field=target_field),
            "test": cls(splits.test_records, splits.circuit_store, target_field=target_field),
        }


def collate_gnn_batch(batch: List[CircuitCandidateData]) -> Batch:
    """Collate a list of CircuitCandidateData into a single batched PyG Data object."""
    return Batch.from_data_list(batch)
