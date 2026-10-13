"""Graph visualizer component for quantum circuit graphs using Plotly."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from qco.circuits.representation import CircuitData
from qco.graphs.circuit_graph import GATE_VOCAB, circuit_to_graph, compute_gate_layers


def render_graph_network_plotly(circuit: CircuitData) -> go.Figure:
    """Create a 2D network diagram of the circuit DAG / dependency graph with Plotly."""
    if len(circuit.gates) == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty Circuit (No Nodes)")
        return fig

    pyg_data = circuit_to_graph(circuit)
    gate_layers = compute_gate_layers(circuit)

    # Node positions: x = DAG layer position, y = - primary qubit index (so q0 is at the top)
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    gate_palette = {
        "H": "#38BDF8",   # Sky blue
        "CX": "#A855F7",  # Purple
        "X": "#F43F5E",   # Rose red
        "Y": "#FB923C",   # Orange
        "Z": "#FBBF24",   # Amber
        "RX": "#34D399",  # Emerald
        "RY": "#2DD4BF",  # Teal
        "RZ": "#22C55E",  # Green
        "I": "#94A3B8",   # Slate
    }

    for idx, gate in enumerate(circuit.gates):
        x = gate_layers[idx]
        y = -float(gate.qubits[0])
        # If two-qubit gate, slightly offset or indicate target
        node_x.append(x)
        node_y.append(y)
        color = gate_palette.get(gate.name, "#6366F1")
        node_color.append(color)

        params_info = f"<br>Params: {gate.params}" if gate.params else ""
        hover_txt = (
            f"<b>Node #{idx}: {gate.name}</b><br>"
            f"Qubits: {gate.qubits}<br>"
            f"Layer: {gate_layers[idx]}"
            f"{params_info}"
        )
        node_text.append(hover_txt)

    # Build Edge Traces (Temporal dependency and entangling lines)
    edge_x = []
    edge_y = []
    edge_index = pyg_data.edge_index.cpu().numpy()

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        edge_x.extend([node_x[src], node_x[dst], None])
        edge_y.extend([node_y[src], node_y[dst], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#64748B"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[f"{g.name}" for g in circuit.gates],
        textposition="top center",
        textfont=dict(color="#F8FAFC", size=11, family="monospace"),
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=22,
            line=dict(width=2, color="#0F172A"),
        ),
        showlegend=False,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="<b>Directed Dependency Graph (Circuit DAG)</b>",
                font=dict(color="#F8FAFC", size=14),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=40, r=40, t=40),
            xaxis=dict(
                title=dict(text="Circuit Layer / Execution Depth", font=dict(color="#94A3B8")),
                showgrid=True,
                zeroline=False,
                gridcolor="#334155",
                tickfont=dict(color="#94A3B8"),
            ),
            yaxis=dict(
                title=dict(text="Qubit Wire", font=dict(color="#94A3B8")),
                showgrid=True,
                zeroline=False,
                gridcolor="#334155",
                tickvals=[-q for q in range(circuit.num_qubits)],
                ticktext=[f"q{q}" for q in range(circuit.num_qubits)],
                tickfont=dict(color="#94A3B8"),
            ),
            plot_bgcolor="#1E293B",
            paper_bgcolor="#0F172A",
            height=320,
        ),
    )
    return fig


def render_graph_inspector_component(circuit: CircuitData) -> None:
    """Render the graph inspection tabs in Streamlit."""
    pyg_data = circuit_to_graph(circuit)
    fig = render_graph_network_plotly(circuit)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔬 View PyG Tensor Embeddings (Nodes & Edges)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Graph Nodes |V|", pyg_data.num_nodes)
        c2.metric("Dependency Edges |E|", pyg_data.edge_index.shape[1])
        c3.metric("Node Feature Dimension", pyg_data.x.shape[1] if pyg_data.num_nodes > 0 else 0)

        t1, t2 = st.tabs(["Node Features (x)", "Edge Connectivity (edge_index)"])
        with t1:
            st.code(f"x shape: {list(pyg_data.x.shape)}\n\nTensor values:\n{pyg_data.x.cpu().numpy()}", language="text")
        with t2:
            st.code(f"edge_index shape: {list(pyg_data.edge_index.shape)}\n\nEdge pairs [source, target]:\n{pyg_data.edge_index.cpu().numpy()}", language="text")
