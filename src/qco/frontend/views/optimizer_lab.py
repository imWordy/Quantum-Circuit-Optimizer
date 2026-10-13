"""Interactive Optimizer Lab View — Student Module 3.

Allows students to:
1. Configure optimization hyperparameters (Optimizer algorithm, iterations, evaluation budget, tolerance).
2. Execute optimization live on the active circuit with real-time feedback.
3. Play back each iteration step with the Interactive Decision Replay slider.
4. Inspect the formal Phase 21 report table and download the full JSON trace report.
"""

from __future__ import annotations

import json
import streamlit as st

from qco.circuits.representation import CircuitData
from qco.experiments.demonstration import (
    DemonstrationReport,
    FinalDemonstrator,
)
from qco.frontend.components.circuit_viewer import render_circuit_component
from qco.frontend.components.metric_cards import render_metric_cards
from qco.frontend.components.trace_timeline import render_trace_timeline_playback
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.models.gnn import CircuitGNN
from qco.optimization.base import OptimizationResult
from qco.optimization.baselines import GreedyOptimizer, RandomOptimizer
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer


def render_optimizer_lab_view() -> None:
    """Render the Interactive Optimizer Lab page."""
    st.markdown("### 🚀 **Interactive Optimizer Lab & Step-by-Step Replay**")
    st.caption(
        "Execute end-to-end circuit optimization. Step through the decision playback to inspect "
        "how GNN guidance prioritizes candidate rewrites and how exact unitary verification guarantees equivalence."
    )

    circuit: CircuitData = st.session_state.current_circuit
    model: CircuitGNN = st.session_state.gnn_model

    # 1. Hyperparameter & Optimizer Configuration Panel
    with st.expander("⚙️ **Optimizer Configuration & Budgets**", expanded=True):
        col_opt, col_iters, col_evals, col_tol = st.columns(4)
        with col_opt:
            opt_choice = st.selectbox(
                "Search Optimizer:",
                options=["GNN-Guided Optimizer (Full Pipeline)", "Greedy Best-Improvement", "Random Baseline"],
                index=0,
            )
        with col_iters:
            max_iters = st.number_input("Max Iterations", min_value=1, max_value=50, value=20)
        with col_evals:
            max_evals = st.number_input("Max Evals / Iteration", min_value=1, max_value=30, value=15)
        with col_tol:
            tol_val = st.selectbox(
                "Unitary Tolerance",
                options=[1e-7, 1e-6, 1e-5],
                index=0,
                format_func=lambda x: f"{x:.1e}",
            )

    # 2. Execution Button
    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_optimization = st.button("🚀 Run Optimization", type="primary")

    if run_optimization:
        with st.spinner("Optimizing circuit through verified rewrite loop..."):
            if opt_choice == "GNN-Guided Optimizer (Full Pipeline)":
                demonstrator = FinalDemonstrator(
                    model=model,
                    max_iterations=int(max_iters),
                    max_evaluations_per_iter=int(max_evals),
                    tolerance=float(tol_val),
                )
                report = demonstrator.run_demonstration(circuit, verbose=False)
                st.session_state.demonstration_report = report
            elif opt_choice == "Greedy Best-Improvement":
                greedy = GreedyOptimizer(max_iterations=int(max_iters), tolerance=float(tol_val))
                res = greedy.optimize(circuit)
                # Wrap in dummy DemonstrationReport for uniform UI display
                report = DemonstrationReport(
                    circuit_id=circuit.circuit_id,
                    num_qubits=circuit.num_qubits,
                    original_cost=res.initial_cost,
                    optimized_cost=res.final_cost,
                    is_equivalent=res.is_equivalent,
                    iterations_run=res.iterations,
                    candidate_evaluations=res.candidate_evaluations,
                    equivalence_checks=res.equivalence_checks,
                    runtime=res.runtime,
                    decision_traces=[],
                    final_circuit=res.final_circuit,
                    original_circuit=circuit,
                )
                st.session_state.demonstration_report = report
            else:
                rand_opt = RandomOptimizer(
                    max_iterations=int(max_iters),
                    max_evaluations_per_iter=int(max_evals),
                    tolerance=float(tol_val),
                )
                res = rand_opt.optimize(circuit)
                report = DemonstrationReport(
                    circuit_id=circuit.circuit_id,
                    num_qubits=circuit.num_qubits,
                    original_cost=res.initial_cost,
                    optimized_cost=res.final_cost,
                    is_equivalent=res.is_equivalent,
                    iterations_run=res.iterations,
                    candidate_evaluations=res.candidate_evaluations,
                    equivalence_checks=res.equivalence_checks,
                    runtime=res.runtime,
                    decision_traces=[],
                    final_circuit=res.final_circuit,
                    original_circuit=circuit,
                )
                st.session_state.demonstration_report = report

        st.success("Optimization completed successfully!")

    st.divider()

    # 3. Results Display
    report: DemonstrationReport | None = st.session_state.demonstration_report
    if report is None:
        st.info("Click **🚀 Run Optimization** above to execute the pipeline on the active circuit.")
        return

    st.markdown("#### 📊 **Optimization Results & Comparison**")

    # Metric Cards showing before vs after deltas
    render_metric_cards(
        cost=report.optimized_cost,
        initial_cost=report.original_cost,
        num_qubits=report.num_qubits,
        circuit_id=report.final_circuit.circuit_id,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Formal Phase 21 Text Summary Card
    st.markdown("##### 📄 **Standard Quantum Optimization Report**")
    st.code(report.format_summary_table(), language="text")

    # Downloadable JSON report
    json_bytes = json.dumps(report.to_dict(), indent=2)
    st.download_button(
        label="💾 Download Optimization Report (JSON)",
        data=json_bytes,
        file_name=f"optimization_report_{report.circuit_id[:8]}.json",
        mime="application/json",
    )

    st.divider()

    # 4. Interactive Step-by-Step Playback
    if report.decision_traces:
        render_trace_timeline_playback(
            initial_circuit=report.original_circuit,
            traces=report.decision_traces,
            final_circuit=report.final_circuit,
        )
    else:
        # Fallback side-by-side view for baselines without detailed step traces
        st.markdown("#### 🔍 **Side-by-Side Circuit Inspection**")
        col_orig, col_opt = st.columns(2)
        with col_orig:
            render_circuit_component(report.original_circuit, title="Original Input Circuit", expanded=True)
        with col_opt:
            render_circuit_component(report.final_circuit, title="Final Optimized Circuit", expanded=True)
