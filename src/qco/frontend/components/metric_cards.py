"""Components for rendering circuit cost metrics and comparisons."""

from __future__ import annotations

from typing import Optional
import streamlit as st

from qco.metrics.circuit_cost import CircuitCost, compute_circuit_cost


def render_metric_cards(
    cost: CircuitCost,
    initial_cost: Optional[CircuitCost] = None,
    num_qubits: int = 2,
    circuit_id: str = "",
) -> None:
    """Render 4 responsive metric summary cards with deltas if initial cost is provided."""
    col1, col2, col3, col4 = st.columns(4)

    # 1. Circuit ID / Qubits
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Circuit ID</div>
                <div class="metric-value" style="font-size: 1.1rem; color: #38BDF8;">{circuit_id[:12]}...</div>
                <div class="metric-delta delta-neutral">{num_qubits} Qubits</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Total Gates
    with col2:
        gate_delta_str = ""
        if initial_cost is not None:
            reduction = initial_cost.total_gates - cost.total_gates
            pct = (reduction / initial_cost.total_gates * 100) if initial_cost.total_gates > 0 else 0.0
            gate_delta_str = f'<div class="metric-delta delta-positive">▼ {reduction} gates ({pct:.1f}%)</div>'
        else:
            gate_delta_str = f'<div class="metric-delta delta-neutral">Single: {cost.single_qubit_gates}</div>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Gates</div>
                <div class="metric-value">{cost.total_gates}</div>
                {gate_delta_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. CNOT Count
    with col3:
        cnot_delta_str = ""
        if initial_cost is not None:
            cnot_red = initial_cost.cnot_count - cost.cnot_count
            cnot_delta_str = f'<div class="metric-delta delta-positive">▼ {cnot_red} CNOTs</div>'
        else:
            cnot_delta_str = '<div class="metric-delta delta-neutral">Entangling gates</div>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CNOT Count</div>
                <div class="metric-value" style="color: #A855F7;">{cost.cnot_count}</div>
                {cnot_delta_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Critical-Path Depth
    with col4:
        depth_delta_str = ""
        if initial_cost is not None:
            depth_red = initial_cost.depth - cost.depth
            depth_delta_str = f'<div class="metric-delta delta-positive">▼ {depth_red} layers</div>'
        else:
            depth_delta_str = '<div class="metric-delta delta-neutral">DAG Critical Path</div>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Circuit Depth</div>
                <div class="metric-value" style="color: #EC4899;">{cost.depth}</div>
                {depth_delta_str}
            </div>
            """,
            unsafe_allow_html=True,
        )
