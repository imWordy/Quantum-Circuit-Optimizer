"""Graph Neural Network Architecture for Quantum Circuit Optimization — Phase 13.

Architecture Specification:
    Circuit Graph (x, edge_index, edge_attr)
          |
    GNN Layers (GCNConv or GATConv with residual connections and ReLU)
          |
    Graph Embedding (Global pooling: mean + max pooling)
          |
    Candidate Features (15-dim vector describing candidate transformation)
          |
    Concatenation [graph_emb, candidate_features]
          |
    MLP Prediction Head (Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear)
          |
    Predicted Gate Reduction (scalar float)

Task:
    Regression: Given (circuit_graph, candidate_proposal), predict expected gate reduction.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

from qco.data.candidate_features import CANDIDATE_FEAT_DIM
from qco.graphs.circuit_graph import VOCAB_SIZE

DEFAULT_NODE_DIM: int = VOCAB_SIZE + 5  # 9 + 5 = 14
DEFAULT_CAND_DIM: int = CANDIDATE_FEAT_DIM  # 15


class CircuitGNN(nn.Module):
    """GNN model predicting the quantitative benefit (gate reduction) of candidate transformations."""

    def __init__(
        self,
        node_dim: int = DEFAULT_NODE_DIM,
        cand_dim: int = DEFAULT_CAND_DIM,
        hidden_dim: int = 64,
        gnn_type: str = "gcn",
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize CircuitGNN model.

        Args:
            node_dim: Input dimension of node feature vectors.
            cand_dim: Input dimension of candidate transformation feature vectors.
            hidden_dim: Hidden dimension for GNN embeddings and MLP head.
            gnn_type: GNN layer type ('gcn' or 'gat').
            num_gnn_layers: Number of message-passing layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.node_dim = node_dim
        self.cand_dim = cand_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type.lower()
        self.num_gnn_layers = num_gnn_layers

        # 1. Input node feature projection
        self.node_in = nn.Linear(node_dim, hidden_dim)

        # 2. GNN message-passing layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            if self.gnn_type == "gat":
                # Single-head GAT preserving hidden_dim
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            else:
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # 3. Global pooling: concatenation of mean_pool and max_pool -> 2 * hidden_dim
        graph_emb_dim = hidden_dim * 2

        # 4. Candidate projection & fusion MLP
        fusion_dim = graph_emb_dim + cand_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def extract_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process graph through GNN layers and pool into a graph-level embedding.

        Args:
            x: Node feature tensor [num_nodes, node_dim].
            edge_index: Edge index tensor [2, num_edges].
            batch: Batch vector [num_nodes] assigning each node to a graph in the batch.

        Returns:
            Tensor of shape [batch_size, 2 * hidden_dim].
        """
        num_nodes = x.size(0)

        # Handle empty graph (0 nodes)
        if num_nodes == 0:
            batch_size = int(batch.max().item() + 1) if (batch is not None and batch.numel() > 0) else 1
            return torch.zeros((batch_size, self.hidden_dim * 2), device=x.device, dtype=x.dtype)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        # Initial node projection
        h = self.node_in(x)

        # GNN message-passing with residual connections
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            h_next = layer(h, edge_index)
            h = norm(h + self.act(h_next))
            h = self.drop(h)

        # Global pooling: combine mean and max representations
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        graph_emb = torch.cat([h_mean, h_max], dim=-1)

        return graph_emb

    def forward(
        self,
        data_or_batch: Union[Data, Batch],
        candidate_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting expected gate reduction.

        Can accept:
        1. A single Batch/Data containing both graph and `candidate_features`.
        2. Graph Data and explicit candidate_features tensor.

        Returns:
            Tensor of shape [batch_size, 1] containing predicted gate reductions.
        """
        if candidate_features is None:
            if not hasattr(data_or_batch, "candidate_features"):
                raise ValueError("data_or_batch must contain 'candidate_features' attribute.")
            cand_feats = data_or_batch.candidate_features
        else:
            cand_feats = candidate_features

        # Determine batch assignment
        batch = getattr(data_or_batch, "batch", None)
        num_graphs = getattr(data_or_batch, "num_graphs", 1)

        # 1. Graph embedding: [num_graphs, 2 * hidden_dim]
        graph_emb = self.extract_graph_embedding(
            x=data_or_batch.x,
            edge_index=data_or_batch.edge_index,
            batch=batch,
        )

        # Ensure candidate features shape matches [num_graphs, cand_dim]
        cand_feats_flat = cand_feats.view(num_graphs, -1)

        # 2. Concatenate [graph_emb, candidate_features]
        combined = torch.cat([graph_emb, cand_feats_flat], dim=-1)

        # 3. Predict gate reduction
        pred = self.head(combined)
        return pred
