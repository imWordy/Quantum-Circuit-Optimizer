"""Candidate Feature Extraction for QCO GNN Interface — Phase 12.

Encodes candidate transformation information into a fixed-length numeric vector:
- Transformation type (one-hot across standard rewrite rules)
- Number of affected gates / indices
- Location / relative position of affected gates in the circuit DAG
- Affected qubits (primary, secondary, count)
- Local gate context before/after the transformation site
- Preliminary cost indicators (preliminary gate count, cnot count, depth)

CRITICAL RULE:
Does NOT include the future target (actual verified gate_reduction, cnot_reduction,
depth_reduction) to prevent test-label leakage into input features.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from qco.circuits.representation import CircuitData, GateOperation
from qco.data.schemas import ObservationRecord
from qco.graphs.circuit_graph import GATE_TO_IDX, VOCAB_SIZE, compute_gate_layers
from qco.metrics.circuit_cost import CircuitCost, compute_circuit_cost
from qco.transformations.candidate_generator import Candidate
from qco.transformations.rules import (
    CNOTCancellation,
    IdentityRemoval,
    RotationCombination,
    SelfInverseCancellation,
    ZeroRotationRemoval,
)

# Standard transformation rule vocabulary for one-hot encoding
RULE_NAMES: List[str] = [
    "SelfInverseCancellation",
    "CNOTCancellation",
    "RotationCombination",
    "ZeroRotationRemoval",
    "IdentityRemoval",
]
RULE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(RULE_NAMES)}
NUM_RULES: int = len(RULE_NAMES)

# Total candidate feature dimension:
# 5 (rule one-hot) + 1 (num_affected_gates) + 1 (first_gate_idx) + 1 (last_gate_idx)
# + 1 (norm_layer) + 1 (num_qubits_affected) + 1 (primary_qubit) + 1 (secondary_qubit)
# + 1 (prelim_gate_count) + 1 (prelim_cnot_count) + 1 (prelim_depth)
# = 5 + 10 = 15 dimensions.
CANDIDATE_FEAT_DIM: int = NUM_RULES + 10


def parse_location_indices(loc_str: str) -> List[int]:
    """Extract integer gate indices from location description string."""
    # Matches patterns like 'indices (1, 2)' or 'index 3'
    m_multi = re.search(r"indices\s*\(([^)]+)\)", loc_str)
    if m_multi:
        try:
            return [int(x.strip()) for x in m_multi.group(1).split(",")]
        except ValueError:
            pass

    m_single = re.search(r"index\s+(\d+)", loc_str)
    if m_single:
        try:
            return [int(m_single.group(1))]
        except ValueError:
            pass

    # Fallback: extract any digits
    digits = re.findall(r"\b\d+\b", loc_str)
    if digits:
        return [int(d) for d in digits]
    return [0]


def extract_candidate_features(
    candidate: Union[Candidate, ObservationRecord],
    parent_circuit: Optional[CircuitData] = None,
) -> torch.Tensor:
    """Extract a 14-dimensional feature vector describing a candidate transformation.

    Guarantees:
    - Never uses target fields (e.g. gate_reduction, cnot_reduction).
    - Features describe ONLY the transformation proposal and its structural site.

    Args:
        candidate: Candidate object or ObservationRecord.
        parent_circuit: Parent CircuitData (required if candidate is ObservationRecord
                        and layer positions are computed; optional otherwise).

    Returns:
        torch.Tensor of shape [CANDIDATE_FEAT_DIM] (float32).
    """
    if isinstance(candidate, Candidate):
        rule_name = candidate.transformation_name
        indices = candidate.transformation_indices
        parent_gates = len(parent_circuit) if parent_circuit is not None else 1
        prelim_gates = float(candidate.preliminary_cost.total_gates)
        prelim_cnot = float(candidate.preliminary_cost.cnot_count)
        prelim_depth = float(candidate.preliminary_cost.depth)
        loc_str = candidate.transformation_location
    elif isinstance(candidate, ObservationRecord):
        rule_name = candidate.transformation_type
        loc_str = candidate.transformation_location
        indices = parse_location_indices(loc_str)
        parent_gates = candidate.parent_gate_count
        prelim_gates = float(candidate.candidate_gate_count)
        prelim_cnot = float(candidate.candidate_cnot_count)
        prelim_depth = float(candidate.candidate_depth)
    else:
        raise TypeError(f"Unsupported candidate type: {type(candidate)}")

    # 1. One-hot transformation rule (5 dims)
    one_hot_rule = [0.0] * NUM_RULES
    if rule_name in RULE_TO_IDX:
        one_hot_rule[RULE_TO_IDX[rule_name]] = 1.0

    # 2. Number of affected gates & index positions normalized by parent length
    num_affected = float(len(indices))
    norm_factor = max(float(parent_gates), 1.0)
    first_idx = float(indices[0]) / norm_factor if indices else 0.0
    last_idx = float(indices[-1]) / norm_factor if indices else 0.0

    # 3. Layer position in circuit DAG
    if parent_circuit is not None and len(parent_circuit) > 0:
        layers = compute_gate_layers(parent_circuit)
        valid_indices = [idx for idx in indices if 0 <= idx < len(layers)]
        if valid_indices:
            avg_layer = float(np.mean([layers[i] for i in valid_indices]))
            max_depth = max(float(max(layers) + 1), 1.0)
            norm_layer = avg_layer / max_depth
        else:
            norm_layer = 0.0
    else:
        norm_layer = first_idx

    # 4. Qubits involved
    affected_qubits: List[int] = []
    if parent_circuit is not None:
        for idx in indices:
            if 0 <= idx < len(parent_circuit):
                affected_qubits.extend(parent_circuit[idx].qubits)
    else:
        # Parse from location string if available (e.g., 'qubit 0')
        q_matches = re.findall(r"qubit\s+(\d+)", loc_str)
        if q_matches:
            affected_qubits.extend(int(q) for q in q_matches)

    unique_qubits = sorted(set(affected_qubits)) if affected_qubits else [0]
    num_qubits_affected = float(len(unique_qubits))
    primary_qubit = float(unique_qubits[0]) if unique_qubits else 0.0
    secondary_qubit = float(unique_qubits[1]) if len(unique_qubits) > 1 else -1.0

    # 5. Preliminary structural cost indicators (NOT actual reductions)
    features = (
        one_hot_rule
        + [
            num_affected,
            first_idx,
            last_idx,
            norm_layer,
            num_qubits_affected,
            primary_qubit,
            secondary_qubit,
            prelim_gates,
            prelim_cnot,
            prelim_depth,
        ]
    )

    tensor_feat = torch.tensor(features, dtype=torch.float32)
    assert tensor_feat.shape[0] == CANDIDATE_FEAT_DIM, (
        f"Feature dim mismatch: expected {CANDIDATE_FEAT_DIM}, got {tensor_feat.shape[0]}"
    )
    return tensor_feat
