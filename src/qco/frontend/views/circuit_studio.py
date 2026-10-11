"""Circuit Studio View — Student Module 1.

Allows students to:
1. Select from curated educational presets (Bell state, GHZ, Teleportation, QFT, Redundancies)
2. Procedurally sample diverse quantum circuits using sliders
3. Upload custom JSON circuit files
4. View live updated wire diagrams, canonical JSON representations, and cost breakdowns
"""

from __future__ import annotations

import json
import streamlit as st

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, SUPPORTED_GATES
from qco.circuits.serialization import deserialize_from_json
from qco.frontend.components.circuit_viewer import render_circuit_component
from qco.frontend.components.metric_cards import render_metric_cards
from qco.frontend.presets import PRESET_CIRCUITS
from qco.metrics.circuit_cost import compute_circuit_cost


def render_circuit_studio_view() -> None:
    """Render the full interactive Circuit Studio page."""
    st.markdown("### 🎨 **Circuit Studio & Interactive Visualizer**")
    st.caption(
        "Compose, generate, or load quantum circuits. Observe how gates are arranged across qubits "
        "and track baseline cost indicators."
    )

    # Circuit Input Source Selection
    input_mode = st.radio(
        "Select Circuit Source:",
        options=["📚 Educational Presets", "🎲 Procedural Generator", "📤 Upload Custom JSON"],
        horizontal=True,
    )

    # 1. Preset Library
    if input_mode == "📚 Educational Presets":
        col_sel, col_desc = st.columns([1, 2])
        with col_sel:
            chosen_preset = st.selectbox(
                "Choose a Preset Circuit:",
                options=list(PRESET_CIRCUITS.keys()),
                index=0,
            )
        with col_desc:
            preset_circuit = PRESET_CIRCUITS[chosen_preset]
            desc = preset_circuit.metadata.get("description", "Educational quantum circuit example")
            st.info(f"**Description**: {desc}\n\n**Qubits**: {preset_circuit.num_qubits} | **Gates**: {len(preset_circuit)}")

        if st.button("Load Preset Circuit", type="primary"):
            st.session_state.current_circuit = preset_circuit
            st.session_state.optimization_result = None
            st.success(f"Loaded '{chosen_preset}' into active workspace!")

    # 2. Procedural Generator
    elif input_mode == "🎲 Procedural Generator":
        with st.form("procedural_generator_form"):
            col_q, col_g, col_seed = st.columns(3)
            with col_q:
                n_qubits = st.slider("Qubits", min_value=2, max_value=5, value=2)
            with col_g:
                n_gates = st.slider("Gate Count", min_value=5, max_value=30, value=12)
            with col_seed:
                gen_seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42)

            allowed = st.multiselect(
                "Allowed Gate Types:",
                options=list(SUPPORTED_GATES.keys()),
                default=["H", "CX", "X", "RZ", "RY"],
            )

            submit_gen = st.form_submit_button("Generate New Random Circuit", type="primary")

            if submit_gen:
                if not allowed:
                    st.error("Please select at least one allowed gate type.")
                else:
                    gen = CircuitGenerator(
                        min_qubits=n_qubits,
                        max_qubits=n_qubits,
                        min_gates=n_gates,
                        max_gates=n_gates,
                        allowed_gates=allowed,
                        seed=int(gen_seed),
                    )
                    new_circ = gen.generate_circuit()
                    st.session_state.current_circuit = new_circ
                    st.session_state.optimization_result = None
                    st.success(f"Generated new circuit with ID: {new_circ.circuit_id[:12]}")

    # 3. Custom Upload
    else:
        uploaded_file = st.file_uploader("Upload Circuit JSON File", type=["json"])
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                loaded_circ = deserialize_from_json(content)
                if st.button("Apply Uploaded Circuit", type="primary"):
                    st.session_state.current_circuit = loaded_circ
                    st.session_state.optimization_result = None
                    st.success("Custom circuit uploaded and loaded successfully!")
            except Exception as e:
                st.error(f"Error parsing circuit JSON: {e}")

    st.divider()

    # Active Circuit Display
    active_circuit: CircuitData = st.session_state.current_circuit
    active_cost = compute_circuit_cost(active_circuit)

    st.markdown("#### 🔍 **Active Circuit Inspection**")
    render_metric_cards(
        cost=active_cost,
        num_qubits=active_circuit.num_qubits,
        circuit_id=active_circuit.circuit_id,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_circuit_component(
        circuit=active_circuit,
        title=f"Circuit Wire Diagram (ID: {active_circuit.circuit_id[:12]})",
        expanded=True,
    )
