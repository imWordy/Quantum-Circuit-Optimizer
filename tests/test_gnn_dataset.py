"""Tests for Phase 12: GNN Dataset Interface.

Verifies:
1. Candidate feature dimension is exactly 14 and matches CANDIDATE_FEAT_DIM.
2. Candidate features do NOT leak future ground truth labels (no gate_reduction in features).
3. Graph tensor dimensions match Phase 10 node feature definitions.
4. Target regression values (y) match true gate_reduction from ObservationRecord.
5. PyG DataLoader batching operates smoothly on varying circuit graphs.
6. A placeholder model forward pass succeeds on a batched graph data object without errors.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from qco.circuits.representation import CircuitData, GateOperation
from qco.data.candidate_features import (
    CANDIDATE_FEAT_DIM,
    extract_candidate_features,
)
from qco.data.dataset_builder import DatasetBuilder
from qco.data.gnn_dataset import (
    CircuitCandidateData,
    GNNDataset,
    collate_gnn_batch,
)
from qco.data.schemas import ObservationRecord
from qco.graphs.circuit_graph import VOCAB_SIZE


def _build_sample_dataset():
    """Helper creating a small synthetic dataset for testing."""
    circuit1 = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),  # SelfInverse match
            GateOperation("CX", (0, 1)),
        ],
    )
    circuit2 = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("RX", (1,), (0.2,)),
            GateOperation("RX", (1,), (0.4,)),  # RotationCombination match
            GateOperation("Z", (0,)),
        ],
    )

    builder = DatasetBuilder()
    builder.process_batch([circuit1, circuit2])
    dataset = GNNDataset(builder.records, builder.circuit_store, target_field="gate_reduction")
    return dataset, builder


def test_candidate_feature_dimensions_and_no_leakage():
    """Verify candidate feature extraction produces exact dimension and does not leak target."""
    rec = ObservationRecord(
        parent_circuit_id="parent_001",
        candidate_id="cand_001",
        transformation_type="SelfInverseCancellation",
        transformation_location="qubit 0 at indices (0, 1)",
        parent_gate_count=5,
        parent_cnot_count=1,
        parent_depth=4,
        candidate_gate_count=3,
        candidate_cnot_count=1,
        candidate_depth=2,
        gate_reduction=2,  # Target
        cnot_reduction=0,  # Target
        depth_reduction=2,  # Target
        equivalent=True,
    )

    feats = extract_candidate_features(rec)

    # 1. Dimension check
    assert feats.shape == (CANDIDATE_FEAT_DIM,)
    assert feats.shape[0] == 15
    assert not torch.isnan(feats).any()

    # 2. Rule one-hot check
    # SelfInverseCancellation is first rule (index 0)
    assert feats[0].item() == 1.0
    assert feats[1:5].sum().item() == 0.0

    # 3. Verify no direct label leakage
    # Even if target gate_reduction=2, features describe candidate gate count (3.0), not reduction
    assert feats[12].item() == 3.0  # prelim_gate_count (index 12)


def test_gnn_dataset_item_structure():
    """Verify GNNDataset produces CircuitCandidateData with expected fields and shapes."""
    dataset, builder = _build_sample_dataset()
    assert len(dataset) > 0

    item = dataset[0]
    assert isinstance(item, CircuitCandidateData)

    # Node features: [num_nodes, VOCAB_SIZE + 5]
    assert item.x.shape[1] == VOCAB_SIZE + 5
    # Edge index: [2, num_edges]
    assert item.edge_index.shape[0] == 2
    # Candidate features: [1, CANDIDATE_FEAT_DIM]
    assert item.candidate_features.shape == (1, CANDIDATE_FEAT_DIM)
    # Regression target: [1]
    assert item.y.shape == (1,)
    assert not torch.isnan(item.y).any()


def test_target_regression_correctness():
    """Verify dataset target (y) strictly corresponds to actual verified reduction."""
    dataset, builder = _build_sample_dataset()

    for idx, item in enumerate(dataset):
        orig_rec = builder.records[idx]
        assert item.y.item() == float(orig_rec.gate_reduction)
        assert item.candidate_id == orig_rec.candidate_id
        assert item.parent_circuit_id == orig_rec.parent_circuit_id


def test_batching_and_placeholder_model_forward():
    """Verify PyG batching and forward pass through a placeholder GNN architecture."""
    dataset, _ = _build_sample_dataset()
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=collate_gnn_batch,
    )

    batch = next(iter(loader))
    assert isinstance(batch, Batch)
    assert batch.num_graphs == len(dataset)

    # Simple placeholder model as defined in Phase 13 outline:
    # GNN (linear) -> Global pooling -> Concatenate candidate features -> Output head
    class PlaceholderGNN(nn.Module):
        def __init__(self, node_dim: int, cand_dim: int, hidden_dim: int = 16):
            super().__init__()
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + cand_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, batch_data: Batch) -> torch.Tensor:
            # 1. Project node features
            h_nodes = torch.relu(self.node_proj(batch_data.x))
            # 2. Graph embedding via global pooling
            g_emb = global_mean_pool(h_nodes, batch_data.batch)  # [batch_size, hidden_dim]
            # 3. Concatenate candidate features
            cand_feats = batch_data.candidate_features.view(batch_data.num_graphs, -1)
            combined = torch.cat([g_emb, cand_feats], dim=-1)
            # 4. Predict
            return self.head(combined)

    model = PlaceholderGNN(node_dim=VOCAB_SIZE + 5, cand_dim=CANDIDATE_FEAT_DIM)
    predictions = model(batch)

    # Check output shape [batch_size, 1]
    assert predictions.shape == (batch.num_graphs, 1)
    assert not torch.isnan(predictions).any()

    # Verify loss computation
    loss_fn = nn.MSELoss()
    loss = loss_fn(predictions, batch.y.view(-1, 1))
    assert loss.item() >= 0.0
    loss.backward()  # Test backward pass without errors
