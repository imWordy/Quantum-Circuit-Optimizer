"""Aggregate-Feature MLP Baseline Model for QCO — Phase 11 (Baseline C).

Extracts aggregate circuit-level statistics to establish an aggregate-feature
baseline for structure-aware vs. aggregate comparison:
- Qubit count
- Total gate count
- CNOT count
- Circuit depth
- Single-qubit gate count
- CNOT ratio (CNOT count / total gates)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Sequence, Union

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import compute_circuit_cost


def extract_aggregate_features(circuit: CircuitData) -> torch.Tensor:
    """Extract 6 aggregate statistics from a quantum circuit into a tensor.

    Features:
    1. Qubit count
    2. Total gate count
    3. CNOT count
    4. Circuit depth
    5. Single-qubit gate count
    6. CNOT ratio (cnot_count / total_gates, 0.0 if total_gates == 0)

    Returns:
        Tensor of shape [6] with float32 values.
    """
    cost = compute_circuit_cost(circuit)
    cnot_ratio = (cost.cnot_count / cost.total_gates) if cost.total_gates > 0 else 0.0

    features = [
        float(circuit.num_qubits),
        float(cost.total_gates),
        float(cost.cnot_count),
        float(cost.depth),
        float(cost.single_qubit_gates),
        float(cnot_ratio),
    ]

    return torch.tensor(features, dtype=torch.float32)


class AggregateFeatureMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) baseline using aggregate circuit statistics."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, input_dim] or [input_dim].

        Returns:
            Tensor of shape [batch_size, output_dim] or [output_dim].
        """
        return self.net(x)
