"""Quantum Circuit Optimizer (QCO) — Student Learning & Experimentation Platform.

Phase 1 Web Application Entrypoint.
"""

from __future__ import annotations

import streamlit as st

from qco.circuits.representation import CircuitData
from qco.frontend.config import (
    apply_custom_css,
    init_session_state,
)
from qco.metrics.circuit_cost import compute_circuit_cost

# 1. Page Configuration
st.set_page_config(
    page_title="Quantum Circuit Optimizer | QCO Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Styling & State Initialization
apply_custom_css()
init_session_state()

# 3. Sidebar Header & Navigation
with st.sidebar:
    st.markdown("## ⚛️ **QCO Platform**")
    st.caption("Deep Learning Lab — Quantum Circuit Optimization with Graph Neural Networks")
    st.divider()

    st.markdown("### 📚 **Student Modules**")
    selected_view = st.radio(
        "Navigation",
        options=[
            "1. Circuit Studio",
            "2. GNN Ranking Inspector",
            "3. Interactive Optimizer Lab",
            "4. Multi-Method Benchmark",
            "5. Exact Verification Hub",
        ],
        index=0,
    )
    st.divider()

    # System Status Indicator
    st.markdown("### ⚙️ **System Status**")
    st.success("● Core Engine Active")
    st.info("● GNN Model Ready (CPU/PyG)")
    st.caption("Verification Authority: Unitary matrix exact equivalence (tol <= 1e-7)")

# 4. Top Header Banner
st.title("⚛️ Quantum Circuit Optimization Laboratory")
st.markdown(
    "Explore how Graph Neural Networks and mathematically rigorous rewrite rules "
    "optimize quantum circuits while preserving exact unitary equivalence."
)
st.divider()

# 5. Active Circuit Snapshot Badge
circuit: CircuitData = st.session_state.current_circuit
cost = compute_circuit_cost(circuit)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Active Circuit ID</div>
            <div class="metric-value" style="font-size: 1.1rem; color: #38BDF8;">{circuit.circuit_id[:12]}...</div>
            <div class="metric-delta delta-neutral">{circuit.num_qubits} Qubits</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Gates</div>
            <div class="metric-value">{cost.total_gates}</div>
            <div class="metric-delta delta-neutral">Single: {cost.single_qubit_gates}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">CNOT Count</div>
            <div class="metric-value" style="color: #A855F7;">{cost.cnot_count}</div>
            <div class="metric-delta delta-neutral">Entangling gates</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Circuit Depth</div>
            <div class="metric-value" style="color: #EC4899;">{cost.depth}</div>
            <div class="metric-delta delta-neutral">DAG Critical Path</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 6. Tab / View Placeholder Router
if selected_view == "1. Circuit Studio":
    from qco.frontend.views.circuit_studio import render_circuit_studio_view
    render_circuit_studio_view()

elif selected_view == "2. GNN Ranking Inspector":
    st.subheader("2. 🧠 GNN Ranking Inspector")
    st.info(
        "In Phase 3, this inspector will visualize circuit graph nodes and edges, "
        "and display the live GNN candidate ranking table."
    )

elif selected_view == "3. Interactive Optimizer Lab":
    st.subheader("3. 🚀 Interactive Optimizer Lab")
    st.info(
        "In Phase 4, students will run GNN-guided optimization with an interactive slider "
        "to replay the exact decision trace step-by-step."
    )

elif selected_view == "4. Multi-Method Benchmark":
    st.subheader("4. 📊 Multi-Method Benchmark Arena")
    st.info(
        "In Phase 5, students will compare Random, Greedy, MLP, and GNN side-by-side with interactive Plotly charts."
    )

elif selected_view == "5. Exact Verification Hub":
    st.subheader("5. 🛡️ Exact Verification Authority Hub")
    st.info(
        "In Phase 6, students will view full unitary matrix heatmaps and verify global phase equivalence proofs."
    )
