"""Components for rendering quantum circuits visually in Streamlit."""

from __future__ import annotations

import json
from typing import Optional
import streamlit as st

from qco.circuits.representation import CircuitData
from qco.circuits.serialization import serialize_to_json


def render_circuit_text(circuit: CircuitData) -> str:
    """Generate Qiskit text wire diagram for a CircuitData instance."""
    try:
        qc = circuit.to_qiskit()
        # Qiskit circuit drawer in UTF-8 text mode
        return str(qc.draw("text"))
    except Exception as e:
        # Graceful fallback: render ASCII sequence table
        lines = [f"Circuit ID: {circuit.circuit_id} ({circuit.num_qubits} qubits)"]
        for idx, g in enumerate(circuit.gates):
            p_str = f"({', '.join(f'{p:.3f}' for p in g.params)})" if g.params else ""
            q_str = f"q{g.qubits}"
            lines.append(f"[{idx:02d}] {g.name}{p_str:<12} on {q_str}")
        return "\n".join(lines)


def render_circuit_component(
    circuit: CircuitData,
    title: str = "Quantum Circuit Wire Diagram",
    show_json_download: bool = True,
    expanded: bool = True,
) -> None:
    """Render interactive multi-view circuit container."""
    with st.expander(title, expanded=expanded):
        tab_diagram, tab_json, tab_table = st.tabs(["⚡ Wire Diagram", "📄 Canonical JSON", "📋 Gate Table"])

        with tab_diagram:
            diagram_text = render_circuit_text(circuit)
            st.markdown(
                f'<div class="circuit-box">{diagram_text}</div>',
                unsafe_allow_html=True,
            )

        with tab_json:
            json_str = serialize_to_json(circuit, indent=2)
            st.code(json_str, language="json")
            if show_json_download:
                st.download_button(
                    label="💾 Download Circuit JSON",
                    data=json_str,
                    file_name=f"circuit_{circuit.circuit_id[:8]}.json",
                    mime="application/json",
                )

        with tab_table:
            gate_data = [
                {
                    "Index": i,
                    "Gate": g.name,
                    "Qubits": ", ".join(map(str, g.qubits)),
                    "Parameters": str([round(p, 4) for p in g.params]) if g.params else "-",
                }
                for i, g in enumerate(circuit.gates)
            ]
            st.dataframe(gate_data, use_container_width=True)

