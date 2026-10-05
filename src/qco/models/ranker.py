"""Candidate Ranking and Ranking Evaluation Metrics for QCO — Phase 15.

Provides:
- CandidateRanker: Uses a trained CircuitGNN to score and rank candidate transformations
  generated from a current quantum circuit.
- RankedCandidate container storing candidate, predicted score, and verification status.
- compute_ranking_metrics: Evaluates ranking quality on unseen circuits using:
  - Spearman rank correlation
  - Top-1 success (whether the #1 predicted candidate is strictly beneficial)
  - Top-k success (whether at least one strictly beneficial candidate is in the top-k)
  - Mean actual improvement among top-k candidates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
import torch
from torch_geometric.data import Batch

from qco.circuits.representation import CircuitData
from qco.data.candidate_features import extract_candidate_features
from qco.data.gnn_dataset import CircuitCandidateData, collate_gnn_batch
from qco.graphs.circuit_graph import circuit_to_graph
from qco.models.gnn import CircuitGNN
from qco.transformations.candidate_generator import Candidate, CandidateGenerator
from qco.utils.logging import setup_logger

logger = setup_logger("qco.models.ranker")


@dataclass
class RankedCandidate:
    """A proposed transformation candidate paired with its GNN predicted score."""

    candidate: Candidate
    predicted_score: float
    rank: int


def compute_ranking_metrics(
    predicted_scores: Sequence[float],
    actual_reductions: Sequence[float],
    k: int = 3,
) -> Dict[str, float]:
    """Compute ranking evaluation metrics comparing model predictions to ground-truth outcomes.

    Metrics:
    - spearman_rho: Spearman rank correlation coefficient.
    - top1_success: 1.0 if top predicted candidate achieves actual reduction > 0, else 0.0.
    - topk_success: 1.0 if any of top-k predicted candidates achieves actual reduction > 0, else 0.0.
    - topk_mean_actual_improvement: Mean actual improvement of top-k candidates.
    - best_possible_improvement: Maximum actual improvement available among all candidates.

    Args:
        predicted_scores: Model predicted benefit scores for candidates.
        actual_reductions: Ground-truth reductions for the same candidates.
        k: Cutoff for top-k metrics.

    Returns:
        Dict with keys: 'spearman_rho', 'spearman_pvalue', 'top1_success',
        'topk_success', 'topk_mean_actual_improvement', 'best_possible_improvement'.
    """
    preds = np.asarray(predicted_scores, dtype=float)
    actuals = np.asarray(actual_reductions, dtype=float)
    n = len(preds)

    if n == 0:
        return {
            "spearman_rho": 0.0,
            "spearman_pvalue": 1.0,
            "top1_success": 0.0,
            "topk_success": 0.0,
            "topk_mean_actual_improvement": 0.0,
            "best_possible_improvement": 0.0,
        }

    # Spearman rank correlation
    if n > 1 and np.std(preds) > 1e-8 and np.std(actuals) > 1e-8:
        rho, pval = scipy.stats.spearmanr(preds, actuals)
        spearman_rho = float(rho) if not np.isnan(rho) else 0.0
        spearman_pval = float(pval) if not np.isnan(pval) else 1.0
    else:
        # Constant predictions or single candidate
        spearman_rho = 0.0
        spearman_pval = 1.0

    # Sort indices by predicted score descending
    sorted_indices = np.argsort(-preds)
    actuals_sorted = actuals[sorted_indices]

    effective_k = min(k, n)
    top1_actual = actuals_sorted[0]
    topk_actuals = actuals_sorted[:effective_k]

    top1_success = 1.0 if top1_actual > 0 else 0.0
    topk_success = 1.0 if np.any(topk_actuals > 0) else 0.0
    topk_mean_improvement = float(np.mean(topk_actuals))
    best_possible = float(np.max(actuals))

    return {
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pval,
        "top1_success": top1_success,
        "topk_success": topk_success,
        "topk_mean_actual_improvement": topk_mean_improvement,
        "best_possible_improvement": best_possible,
    }


class CandidateRanker:
    """Uses a trained CircuitGNN to score and rank proposed candidate transformations.

    Critical principle:
    Prediction does NOT equal acceptance. Every candidate ranked by this model
    must still be independently verified for validity, equivalence, and cost before adoption.
    """

    def __init__(
        self,
        model: CircuitGNN,
        candidate_generator: Optional[CandidateGenerator] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize candidate ranker.

        Args:
            model: Trained CircuitGNN model.
            candidate_generator: CandidateGenerator instance (defaults to standard rules).
            device: Computing device ('cpu' or 'cuda').
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.generator = candidate_generator or CandidateGenerator()

    def rank_candidates(
        self,
        circuit: CircuitData,
        candidates: Optional[Sequence[Candidate]] = None,
    ) -> List[RankedCandidate]:
        """Generate (if not provided), score, and rank candidates by predicted benefit descending.

        Args:
            circuit: Current quantum circuit.
            candidates: Optional pre-generated candidates. If None, generated automatically.

        Returns:
            List of RankedCandidate objects ordered from highest to lowest predicted benefit.
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(circuit)

        if not candidates:
            return []

        # Convert parent circuit to graph (Phase 10)
        graph_data = circuit_to_graph(circuit)

        # Build PyG Data objects for batching
        items: List[CircuitCandidateData] = []
        for cand in candidates:
            cand_feats = extract_candidate_features(cand, parent_circuit=circuit)
            item = CircuitCandidateData(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr,
                candidate_features=cand_feats.unsqueeze(0),
                num_nodes=graph_data.num_nodes,
                candidate_id=cand.candidate_id,
            )
            items.append(item)

        batch = collate_gnn_batch(items).to(self.device)

        # Run GNN forward pass
        with torch.no_grad():
            preds = self.model(batch).view(-1).cpu().numpy()

        # Combine candidates with scores and sort descending
        scored = list(zip(candidates, preds))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        ranked: List[RankedCandidate] = []
        for rank_idx, (cand, score) in enumerate(scored, start=1):
            ranked.append(
                RankedCandidate(
                    candidate=cand,
                    predicted_score=float(score),
                    rank=rank_idx,
                )
            )

        logger.debug(
            f"Circuit {circuit.circuit_id}: ranked {len(ranked)} candidates. "
            f"Top score: {ranked[0].predicted_score:.3f} ({ranked[0].candidate.transformation_name})"
        )
        return ranked
