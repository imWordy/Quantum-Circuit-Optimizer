"""Interactive timeline and trace playback components for optimization."""

from __future__ import annotations

from typing import List, Optional
import streamlit as st

from qco.circuits.representation import CircuitData
from qco.experiments.demonstration import IterationDecisionTrace
from qco.frontend.components.circuit_viewer import render_circuit_component


def render_trace_timeline_playback(
    initial_circuit: CircuitData,
    traces: List[IterationDecisionTrace],
    final_circuit: CircuitData,
) -> None:
    """Render interactive playback scrubber for optimization decisions."""
    if not traces:
        st.info("Optimization terminated immediately (circuit was already in a local minimum for the rule set).")
        return

    st.markdown("#### ⏯️ **Step-by-Step Decision Playback**")
    st.caption("Drag the slider to scrub through each iteration of the decision pipeline.")

    # Timeline Slider
    max_steps = len(traces)
    step_idx = st.slider(
        "Iteration Step:",
        min_value=0,
        max_value=max_steps,
        value=max_steps,
        format="Step %d",
    )

    if step_idx == 0:
        st.markdown("##### 📍 **Initial State (Before Optimization)**")
        st.info("Starting quantum circuit prior to candidate generation and GNN ranking.")
        render_circuit_component(
            initial_circuit,
            title=f"Initial Circuit Wire Diagram (ID: {initial_circuit.circuit_id[:12]})",
            expanded=True,
        )
    else:
        trace = traces[step_idx - 1]
        st.markdown(f"##### 📍 **Iteration {trace.iteration}: {trace.transformation_applied}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gate Count", f"{trace.gate_count_before} → {trace.gate_count_after}", delta=f"{trace.gate_count_after - trace.gate_count_before} gates")
        c2.metric("CNOT Count", f"{trace.cnot_count_before} → {trace.cnot_count_after}", delta=f"{trace.cnot_count_after - trace.cnot_count_before} CNOTs")
        c3.metric("Critical Depth", f"{trace.depth_before} → {trace.depth_after}", delta=f"{trace.depth_after - trace.depth_before} layers")
        c4.markdown(
            f"""
            <div style="padding-top: 10px;">
                <span class="badge-verified">✓ Unitary Verified</span>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Matrix dist: {trace.matrix_distance:.2e}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View Decision & Candidate Ranking Details for this Step", expanded=False):
            st.markdown(f"**Applied Transformation**: `{trace.transformation_applied}` at `{trace.transformation_location}`")
            st.markdown(f"**Status / Reason**: {trace.reason}")
            st.markdown(f"**Candidate Pool Size**: {trace.num_candidates_generated} candidates evaluated by GNN")

            if trace.ranked_candidates:
                st.caption("Top 5 candidates prioritized by GNN at this iteration:")
                top_data = [
                    {
                        "Rank": c.get("rank"),
                        "Rule": c.get("rule"),
                        "Location": c.get("location"),
                        "Predicted Benefit": f"{c.get('predicted_gate_reduction', 0.0):+.4f}",
                    }
                    for c in trace.ranked_candidates
                ]
                st.dataframe(top_data, use_container_width=True)

        # Show resulting circuit at this step
        circ_to_show = trace.circuit_after or (final_circuit if step_idx == max_steps else initial_circuit)
        render_circuit_component(
            circ_to_show,
            title=f"Circuit After Step {trace.iteration} (ID: {circ_to_show.circuit_id[:12]})",
            expanded=True,
        )
