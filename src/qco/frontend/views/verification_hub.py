"""Equivalence Verification Authority Hub View for QCO.

Pedagogical deep dive into exact unitary equivalence, global phase factor invariance,
and residual error proofs for student learning.
"""

from __future__ import annotations

import copy
import numpy as np
import streamlit as st

from qco.circuits.representation import CircuitData, GateOperation
from qco.frontend.components.unitary_viewer import (
    create_product_distance_heatmap,
    create_unitary_heatmap,
)
from qco.verification.equivalence import check_unitary_equivalence
from qco.verification.simulator import compute_unitary


def render_verification_hub_view() -> None:
    """Render Module 5: Exact Verification Authority Hub."""
    st.subheader("🛡️ Module 5: Exact Verification Authority Hub")
    st.markdown(
        "Explore the mathematical proof mechanism of quantum circuit equivalence. "
        "In quantum computing, two circuits $U_A$ and $U_B$ execute identical physical operations "
        "if and only if they differ by at most an unobservable global phase factor: "
        "$$U_A = e^{i\\phi} U_B \\iff U_A U_B^\\dagger = e^{i\\phi} I$$"
    )
    st.divider()

    active_circuit: CircuitData = st.session_state.current_circuit

    # 1. Comparison Target Selector
    st.markdown("### 1. Select Circuit for Equivalence Verification")
    col_tgt1, col_tgt2 = st.columns([2, 1])

    with col_tgt1:
        target_options = ["Optimized Circuit from Optimizer Lab", "Active Circuit (Self-identity test)"]
        if "demonstration_report" in st.session_state and st.session_state.demonstration_report is not None:
            default_idx = 0
        else:
            target_options = ["Active Circuit (Self-identity test)"]
            default_idx = 0

        target_selection = st.radio(
            "Comparison Candidate:",
            options=target_options,
            index=default_idx,
            horizontal=True,
        )

    with col_tgt2:
        tol_exp = st.slider(
            "Equivalence Tolerance (10^x):",
            min_value=-12,
            max_value=-3,
            value=-7,
            step=1,
            help="Threshold below which max matrix distance is considered equivalent.",
        )
        tolerance = 10.0**tol_exp
        st.caption(f"Numerical Tolerance: `ε = {tolerance:.1e}`")

    # Determine candidate circuit B
    if (
        target_selection == "Optimized Circuit from Optimizer Lab"
        and "demonstration_report" in st.session_state
        and st.session_state.demonstration_report is not None
    ):
        report = st.session_state.demonstration_report
        # Support both .final_circuit and .optimized_circuit
        circuit_b = getattr(report, "final_circuit", getattr(report, "optimized_circuit", None))
        if circuit_b is None:
            circuit_b = copy.deepcopy(active_circuit)
        candidate_label = "Optimized Circuit"
    else:
        circuit_b = copy.deepcopy(active_circuit)
        candidate_label = "Identical Circuit Copy"

    # 2. Interactive Perturbation Tool (Educational Failure Injection)
    st.markdown("### 2. 🧪 Interactive Perturbation Injector (Pedagogical Test)")
    st.markdown(
        "To test how rigorously the verification authority detects invalid rewrites or non-equivalent mutations, "
        "inject an artificial perturbation into Candidate $U_B$:"
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
    with col_p1:
        inject_phase = st.checkbox("Inject Global Phase e^(i·π/4)", value=False, help="Multiplies U_B by e^(i·π/4); should PASS if global phase is ignored!")
    with col_p2:
        inject_bug = st.checkbox("Inject Perturbation Error (Add X Gate)", value=False, help="Adds an extra X gate to qubit 0; MUST FAIL equivalence!")
    with col_p3:
        ignore_global_phase = st.checkbox("Allow Global Phase Invariance", value=True, help="When enabled, U_A U_B† = e^(iφ) I is accepted.")

    # Apply perturbations to circuit_b or its unitary
    circuit_b_modified = copy.deepcopy(circuit_b)
    if inject_bug:
        # Add an X gate on qubit 0
        new_gates = list(circuit_b_modified.gates) + [GateOperation(name="x", qubits=[0], params=[])]
        circuit_b_modified = CircuitData(
            circuit_id=f"{circuit_b.circuit_id}_perturbed",
            num_qubits=circuit_b.num_qubits,
            gates=new_gates,
        )

    # 3. Unitary Computation & Verification Authority
    if active_circuit.num_qubits > 6:
        st.warning(f"⚠️ Circuit has {active_circuit.num_qubits} qubits (2^{active_circuit.num_qubits} = {2**active_circuit.num_qubits} matrix dim). Unitary calculation may take a moment.")

    with st.spinner("Computing unitary matrices and evaluating equivalence..."):
        u_a = compute_unitary(active_circuit)
        u_b = compute_unitary(circuit_b_modified)

        if inject_phase:
            u_b = u_b * np.exp(1j * np.pi / 4.0)

        # Mathematical Equivalence Proof
        is_equiv, phase_angle, matrix_dist = check_unitary_equivalence(
            u_a,
            u_b,
            tolerance=tolerance,
            ignore_global_phase=ignore_global_phase,
        )

    # 4. Verification Certificate
    st.divider()
    st.markdown("### 3. 📜 Quantum Equivalence Verification Certificate")

    cert_col1, cert_col2, cert_col3 = st.columns([1.5, 1, 1])

    with cert_col1:
        if is_equiv:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10B981; border-radius: 12px; padding: 1.2rem; text-align: center;">
                    <div style="font-size: 2.2rem;">✅</div>
                    <h3 style="color: #10B981; margin: 0.3rem 0;">VERIFIED EQUIVALENT</h3>
                    <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0;">Circuit A and Candidate B represent identical quantum unitary operations.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #EF4444; border-radius: 12px; padding: 1.2rem; text-align: center;">
                    <div style="font-size: 2.2rem;">❌</div>
                    <h3 style="color: #EF4444; margin: 0.3rem 0;">EQUIVALENCE REJECTED</h3>
                    <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0;">Residual matrix distance exceeded numerical tolerance threshold.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with cert_col2:
        st.metric(
            label="Max Matrix Distance ||U_A U_B† - e^(iφ)I||",
            value=f"{matrix_dist:.2e}",
            delta=f"Threshold: {tolerance:.1e}",
            delta_color="normal" if is_equiv else "inverse",
        )
        st.metric(
            label="Matrix Dimension",
            value=f"{u_a.shape[0]} × {u_a.shape[1]}",
            delta=f"2^{active_circuit.num_qubits} states",
            delta_color="off",
        )

    with cert_col3:
        phase_deg = np.degrees(phase_angle) if phase_angle is not None else 0.0
        st.metric(
            label="Detected Global Phase (φ)",
            value=f"{phase_angle:.4f} rad" if phase_angle is not None else "N/A",
            delta=f"{phase_deg:.1f}°" if phase_angle is not None else "N/A",
            delta_color="off",
        )
        st.metric(
            label="Verification Rule",
            value="Unitary Operator" if ignore_global_phase else "Exact Matrix Match",
            delta="Physics-Preserving",
            delta_color="off",
        )

    # 5. Step-by-Step Mathematical Proof Breakdown
    st.divider()
    st.markdown("### 4. 🧮 Step-by-Step Mathematical Proof Inspector")

    st.markdown(
        r"""
        1. **Compute Circuit Unitaries**:
           $U_A$ is calculated from the DAG representation of the active circuit. $U_B$ is calculated from the candidate rewrite.
        2. **Form Unitary Product**:
           $$P = U_A U_B^\dagger$$
           If $U_A$ and $U_B$ are equivalent up to phase $\phi$, then $P = e^{i\phi} I$.
        3. **Estimate Global Phase Factor $e^{i\phi}$**:
           The diagonal elements of $P$ are inspected. The candidate phase factor is $\lambda = P_{kk} / |P_{kk}|$.
        4. **Compute Residual Error Matrix**:
           $$R = |P - \lambda I|$$
        5. **Certificate Criterion**:
           $$\max_{i,j} R_{ij} \le \varepsilon \quad (\text{Current: } """
        + f"{matrix_dist:.2e} \\le {tolerance:.1e})"
    )

    # 6. Interactive Unitary Heatmap Inspection
    st.divider()
    st.markdown("### 5. 🗺️ Unitary Matrix Heatmaps")

    col_view_opt, _ = st.columns([2, 2])
    with col_view_opt:
        heatmap_mode = st.radio(
            "Heatmap Representation:",
            options=["Real & Imaginary Parts", "Magnitude & Phase"],
            horizontal=True,
        )
    mode_arg = "real_imag" if heatmap_mode == "Real & Imaginary Parts" else "magnitude_phase"

    tab_ua, tab_ub, tab_residual = st.tabs(["Active Circuit Unitary (U_A)", f"Candidate Unitary (U_B: {candidate_label})", "Residual Distance |U_A U_B† - e^(iφ)I|"])

    with tab_ua:
        fig_ua = create_unitary_heatmap(u_a, title="Active Circuit Unitary U_A", mode=mode_arg)
        st.plotly_chart(fig_ua, use_container_width=True)

    with tab_ub:
        fig_ub = create_unitary_heatmap(u_b, title=f"Candidate Unitary U_B ({candidate_label})", mode=mode_arg)
        st.plotly_chart(fig_ub, use_container_width=True)

    with tab_residual:
        n = u_a.shape[0]
        u_prod = u_a @ u_b.conj().T
        if phase_angle is not None and ignore_global_phase:
            expected_prod = np.exp(1j * phase_angle) * np.eye(n, dtype=complex)
        else:
            expected_prod = np.eye(n, dtype=complex)

        fig_res = create_product_distance_heatmap(
            u_prod,
            expected_prod,
            title="Residual Error Matrix: |U_A U_B† - e^(iφ) I|",
        )
        st.plotly_chart(fig_res, use_container_width=True)

        if not is_equiv:
            st.error(
                f"🚨 **Equivalence Violation Detected!** The maximum entry in the residual matrix is "
                f"**{matrix_dist:.4e}**, which exceeds tolerance **{tolerance:.1e}**."
            )
        else:
            st.success(
                f"✨ **Exact Match Confirmed!** All entries in the residual matrix are below "
                f"tolerance ({matrix_dist:.2e} <= {tolerance:.1e})."
            )
