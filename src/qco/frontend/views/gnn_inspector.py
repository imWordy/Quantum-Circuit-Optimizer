"""GNN Ranking Inspector View — Student Module 2.

Allows students to:
1. Visualize the structural graph representation of the active quantum circuit (nodes, edges, layers).
2. Inspect the procedural candidate rewrite rules proposed for the circuit.
3. Observe how the trained CircuitGNN scores and prioritizes candidate transformations in real time.
4. Compare GNN predicted scores against preliminary costs and rule expectations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qco.circuits.representation import CircuitData
from qco.frontend.components.graph_viewer import render_graph_inspector_component
from qco.frontend.components.metric_cards import render_metric_cards
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.models.gnn import CircuitGNN
from qco.models.ranker import CandidateRanker, RankedCandidate
from qco.transformations.candidate_generator import CandidateGenerator


def render_gnn_inspector_view() -> None:
    """Render the GNN Graph & Candidate Ranking Inspector page."""
    st.markdown("### 🧠 **GNN Candidate Ranking & Graph Inspector**")
    st.caption(
        "Observe how the circuit is mapped into a Directed Acyclic Graph (DAG) for PyTorch Geometric, "
        "and see how the Graph Neural Network prioritizes candidate transformation rules."
    )

    circuit: CircuitData = st.session_state.current_circuit
    model: CircuitGNN = st.session_state.gnn_model

    # 1. Structural Graph Visualization
    st.markdown("#### 1. 🕸️ **Structural Graph Representation (PyTorch Geometric)**")
    render_graph_inspector_component(circuit)

    st.divider()

    # 2. Candidate Proposal & GNN Ranking
    st.markdown("#### 2. 🎯 **Real-Time GNN Candidate Ranking**")
    st.markdown(
        "The **Candidate Generator** systematically detects all valid pattern match locations. "
        "The **CircuitGNN** processes the graph structure + candidate feature vectors to predict the quantitative gate reduction."
    )

    ranker = CandidateRanker(model=model)
    ranked_candidates = ranker.rank_candidates(circuit)

    if not ranked_candidates:
        st.warning("No applicable transformation rewrite rules found for this circuit. It is currently at a local minimum for the rule set.")
        return

    # Summary metrics of candidate pool
    c1, c2, c3 = st.columns(3)
    c1.metric("Generated Candidates", len(ranked_candidates))
    highest_pred = max(rc.predicted_score for rc in ranked_candidates)
    c2.metric("Highest GNN Predicted Reduction", f"{highest_pred:+.3f} gates")
    unique_rules = len(set(rc.candidate.transformation_name for rc in ranked_candidates))
    c3.metric("Applicable Rule Types", unique_rules)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ranking Table
    ranking_data = []
    for rc in ranked_candidates:
        cand = rc.candidate
        ranking_data.append(
            {
                "Rank": rc.rank,
                "Rule Name": cand.transformation_name,
                "Location / Site": cand.transformation_location,
                "GNN Predicted Benefit": f"{rc.predicted_score:+.4f}",
                "Prelim Gate Count": cand.preliminary_cost.total_gates,
                "Prelim CNOT Count": cand.preliminary_cost.cnot_count,
                "Prelim Depth": cand.preliminary_cost.depth,
                "Candidate ID": cand.candidate_id[:8],
            }
        )

    df_rank = pd.DataFrame(ranking_data)
    st.dataframe(
        df_rank,
        use_container_width=True,
        hide_index=True,
    )

    # Educational Drilldown on Top Candidate
    top_cand = ranked_candidates[0]
    with st.expander(f"🔍 Educational Inspection: Top-Ranked Candidate (#{top_cand.rank} - {top_cand.candidate.transformation_name})", expanded=True):
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown(f"**Applied Rule**: `{top_cand.candidate.transformation_name}`")
            st.markdown(f"**Target Site**: `{top_cand.candidate.transformation_location}`")
            st.markdown(f"**GNN Prediction Score**: `{top_cand.predicted_score:+.4f}`")
            st.caption("Note: Under the core verification protocol, this prediction serves purely to prioritize search order. Exact unitary equivalence must still be verified before acceptance.")
        with c_right:
            init_cost = compute_circuit_cost(circuit)
            st.markdown(
                f"**Parent Cost**: {init_cost.total_gates} gates (CNOT: {init_cost.cnot_count}, Depth: {init_cost.depth})<br>"
                f"**Proposed Cost**: {top_cand.candidate.preliminary_cost.total_gates} gates (CNOT: {top_cand.candidate.preliminary_cost.cnot_count}, Depth: {top_cand.candidate.preliminary_cost.depth})<br>"
                f"**Expected Reduction**: {init_cost.total_gates - top_cand.candidate.preliminary_cost.total_gates} gates",
                unsafe_allow_html=True,
            )
