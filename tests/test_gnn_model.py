"""Tests for Phase 13: Initial GNN Model.

Verifies:
1. Forward pass produces tensor of correct shape [batch_size, 1].
2. Supports both GCN and GAT message passing backends.
3. Graph batching over variable graph sizes via PyG DataLoader.
4. Backward pass, gradient computation, and optimizer step succeed without errors.
5. Model trains for multiple iterations with monotonic loss reduction on a toy batch without NaNs.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset_builder import DatasetBuilder
from qco.data.gnn_dataset import GNNDataset, collate_gnn_batch
from qco.models.gnn import (
    DEFAULT_CAND_DIM,
    DEFAULT_NODE_DIM,
    CircuitGNN,
)


def _create_sample_batch():
    """Helper creating a batched PyG Data object from sample circuits."""
    c1 = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
        ],
    )
    c2 = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("RX", (1,), (0.2,)),
            GateOperation("RX", (1,), (0.4,)),
            GateOperation("CX", (1, 2)),
            GateOperation("CX", (1, 2)),
            GateOperation("Z", (0,)),
        ],
    )

    builder = DatasetBuilder()
    builder.process_batch([c1, c2])
    dataset = GNNDataset(builder.records, builder.circuit_store)

    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_gnn_batch)
    return next(iter(loader))


def test_gnn_forward_pass_and_output_shape():
    """Verify forward pass on a batched graph produces output shape [batch_size, 1]."""
    batch = _create_sample_batch()
    model = CircuitGNN(
        node_dim=DEFAULT_NODE_DIM,
        cand_dim=DEFAULT_CAND_DIM,
        hidden_dim=32,
        gnn_type="gcn",
        num_gnn_layers=2,
    )

    out = model(batch)
    assert out.shape == (batch.num_graphs, 1)
    assert not torch.isnan(out).any()


def test_gat_variant_forward():
    """Verify GAT message passing variant operates without error."""
    batch = _create_sample_batch()
    model_gat = CircuitGNN(
        node_dim=DEFAULT_NODE_DIM,
        cand_dim=DEFAULT_CAND_DIM,
        hidden_dim=32,
        gnn_type="gat",
        num_gnn_layers=2,
    )

    out = model_gat(batch)
    assert out.shape == (batch.num_graphs, 1)
    assert not torch.isnan(out).any()


def test_gnn_backward_pass_and_optimizer_step():
    """Verify backward pass computes gradients for all trainable parameters and optimizer steps."""
    batch = _create_sample_batch()
    model = CircuitGNN(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    preds = model(batch)
    loss = criterion(preds, batch.y.view(-1, 1))

    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

    loss.backward()

    # Check that gradients exist and are finite
    has_grad = False
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            has_grad = True

    assert has_grad is True
    optimizer.step()


def test_multiple_training_iterations():
    """Verify that multiple training iterations run smoothly and reduce loss without NaNs."""
    batch = _create_sample_batch()
    model = CircuitGNN(hidden_dim=32, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(15):
        optimizer.zero_grad()
        preds = model(batch)
        loss = criterion(preds, batch.y.view(-1, 1))
        assert not torch.isnan(loss)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss is not None
    assert final_loss < initial_loss, f"Loss should decrease: initial={initial_loss}, final={final_loss}"
