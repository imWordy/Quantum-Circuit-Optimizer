"""Frontend configuration, state management, and custom styles."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st
import torch

from qco.circuits.representation import CircuitData, GateOperation
from qco.models.gnn import CircuitGNN


CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-positive {
        color: #10B981;
    }
    .delta-neutral {
        color: #64748B;
    }
    
    /* Code/Circuit box */
    .circuit-box {
        font-family: 'Courier New', Courier, monospace;
        background-color: #090D16;
        color: #38BDF8;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 12px;
        overflow-x: auto;
        white-space: pre;
    }

    /* Verification Badge */
    .badge-verified {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-unverified {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
"""


def apply_custom_css() -> None:
    """Inject custom responsive stylesheet into Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_default_sample_circuit() -> CircuitData:
    """Provide a reliable default quantum circuit with redundant cancellations."""
    gates = [
        GateOperation("H", (0,)),
        GateOperation("H", (0,)),
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (0, 1)),
        GateOperation("RZ", (0,), params=[0.785398]),
        GateOperation("RZ", (0,), params=[-0.785398]),
        GateOperation("X", (1,)),
    ]
    return CircuitData(num_qubits=2, gates=gates)


@st.cache_resource
def load_default_gnn_model() -> CircuitGNN:
    """Cache and load default CircuitGNN model for fast inference."""
    model = CircuitGNN(hidden_dim=64, gnn_type="gcn")
    model.eval()
    return model


def init_session_state() -> None:
    """Initialize persistent Streamlit session state keys."""
    if "current_circuit" not in st.session_state:
        st.session_state.current_circuit = get_default_sample_circuit()
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None
    if "demonstration_report" not in st.session_state:
        st.session_state.demonstration_report = None
    if "gnn_model" not in st.session_state:
        st.session_state.gnn_model = load_default_gnn_model()
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Circuit Studio"
