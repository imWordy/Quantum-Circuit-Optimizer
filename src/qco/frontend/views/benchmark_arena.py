"""Multi-Method Benchmark Arena & Ablation View — Student Module 4.

Allows students to:
1. Select or generate a controlled test suite of quantum circuits.
2. Run side-by-side comparative experiments across all 4 optimizers:
   - Random Optimizer (Baseline A)
   - Greedy Best-Improvement (Baseline B)
   - Aggregate-Feature MLP (Baseline C)
   - GNN-Guided Optimizer (Core Method D)
3. Conduct component ablations:
   - Structural GNN vs Aggregate MLP vs Non-ML baselines (RQ2 / RQ4).
   - Transformation rule subsets (Clifford cancellation vs Continuous rotations vs Full).
4. Visualize comparative Plotly charts (Gate reduction, Search efficiency, Runtime trade-offs).
5. Download raw benchmark results as CSV.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData
from qco.experiments.ablation import AblationFramework, AblationStudyResult
from qco.experiments.benchmark import BenchmarkHarness, BenchmarkRecord
from qco.frontend.components.benchmark_plots import (
    render_efficiency_scatter_plot,
    render_reduction_comparison_bar,
    render_runtime_bar_plot,
)
from qco.frontend.presets import PRESET_CIRCUITS
from qco.models.gnn import CircuitGNN
from qco.models.mlp_baseline import AggregateFeatureMLP
from qco.optimization import (
    GreedyOptimizer,
    MLPOptimizer,
    RandomOptimizer,
)
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer
from qco.optimization.mlp_optimizer import MLPOptimizer
from qco.transformations.candidate_generator import CandidateGenerator
from qco.transformations.rules import (
    CNOTCancellation,
    IdentityRemoval,
    RotationCombination,
    SelfInverseCancellation,
    ZeroRotationRemoval,
    get_default_rules,
)


def render_benchmark_arena_view() -> None:
    """Render the full Multi-Method Benchmark Arena page."""
    st.markdown("### 📊 **Multi-Method Benchmark Arena & Ablations**")
    st.caption(
        "Execute controlled comparative experiments across Random, Greedy, MLP, and GNN optimizers. "
        "Investigate whether structural graph neural representations provide significant search advantages over aggregate statistics."
    )

    gnn_model: CircuitGNN = st.session_state.gnn_model

    # Experiment Mode Selection
    mode = st.radio(
        "Select Experiment Mode:",
        options=[
            "1. 4-Way Comparative Benchmark (Random vs Greedy vs MLP vs GNN)",
            "2. Transformation Rule Subset Ablation (Clifford vs Rotation vs All)",
        ],
        horizontal=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. 4-Way Comparative Benchmark
    if mode.startswith("1."):
        with st.expander("⚙️ **Benchmark Test Suite & Search Budget**", expanded=True):
            c_source, c_iters, c_evals = st.columns([2, 1, 1])
            with c_source:
                suite_choice = st.selectbox(
                    "Test Circuit Suite:",
                    options=[
                        "Educational Presets (5 Circuits)",
                        "Synthetic Benchmark (5 Random Circuits, 2-3 Qubits, 10-15 Gates)",
                        "Active Workspace Circuit Only",
                    ],
                    index=0,
                )
            with c_iters:
                max_iters = st.number_input("Max Iterations", min_value=1, max_value=30, value=12)
            with c_evals:
                max_evals = st.number_input("Max Evals / Iter", min_value=1, max_value=20, value=10)

        if st.button("🚀 Run 4-Way Comparative Benchmark", type="primary"):
            # Prepare test circuit set
            if suite_choice == "Educational Presets (5 Circuits)":
                test_circuits = list(PRESET_CIRCUITS.values())
            elif suite_choice == "Active Workspace Circuit Only":
                test_circuits = [st.session_state.current_circuit]
            else:
                gen = CircuitGenerator(min_qubits=2, max_qubits=3, min_gates=10, max_gates=15, seed=42)
                test_circuits = [gen.generate_circuit() for _ in range(5)]

            with st.spinner(f"Running controlled benchmark across {len(test_circuits)} circuits with 4 optimizers..."):
                # Instantiate all 4 optimizers with identical budget
                opt_rand = RandomOptimizer(max_iterations=int(max_iters), max_evaluations_per_iter=int(max_evals))
                opt_greedy = GreedyOptimizer(max_iterations=int(max_iters))
                mlp_model = AggregateFeatureMLP(hidden_dim=32)
                opt_mlp = MLPOptimizer(model=mlp_model, max_iterations=int(max_iters), max_evaluations_per_iter=int(max_evals))
                opt_gnn = GNNGuidedOptimizer(model=gnn_model, max_iterations=int(max_iters), max_evaluations_per_iter=int(max_evals))

                harness = BenchmarkHarness(optimizers=[opt_rand, opt_greedy, opt_mlp, opt_gnn])
                raw_df = harness.run_benchmark(test_circuits)
                summary_df = harness.compute_summary_table()

                st.session_state["benchmark_raw_df"] = raw_df
                st.session_state["benchmark_summary_df"] = summary_df

            st.success("Comparative benchmark completed!")

        # Display results if available
        if "benchmark_summary_df" in st.session_state and not st.session_state["benchmark_summary_df"].empty:
            summary_df: pd.DataFrame = st.session_state["benchmark_summary_df"]
            raw_df: pd.DataFrame = st.session_state["benchmark_raw_df"]

            st.markdown("#### 🏆 **Aggregated Benchmark Leaderboard**")
            st.dataframe(
                summary_df.style.format(
                    {
                        "avg_gate_reduction": "{:.2f}",
                        "avg_cnot_reduction": "{:.2f}",
                        "avg_depth_reduction": "{:.2f}",
                        "avg_evaluations": "{:.1f}",
                        "avg_checks": "{:.1f}",
                        "avg_runtime_sec": "{:.4f} s",
                    }
                ),
                use_container_width=True,
            )

            csv_data = raw_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Benchmark Observations (CSV)",
                data=csv_data,
                file_name="quantum_benchmark_observations.csv",
                mime="text/csv",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 **Interactive Comparative Analysis**")
            c_bar, c_scatter = st.columns([1, 1])
            with c_bar:
                st.plotly_chart(render_reduction_comparison_bar(summary_df), use_container_width=True)
            with c_scatter:
                st.plotly_chart(render_efficiency_scatter_plot(raw_df), use_container_width=True)

            st.plotly_chart(render_runtime_bar_plot(summary_df), use_container_width=True)

    # 2. Transformation Rule Subset Ablation
    else:
        st.markdown("#### 🔬 **Transformation Rule Subset Ablation**")
        st.caption(
            "Isolate the contribution of distinct algebraic rewrite classes: "
            "Clifford cancellation rules vs Continuous rotation combination vs All combined."
        )

        circuits_for_ablation = [PRESET_CIRCUITS["Redundant Cancellations (Benchmark)"]]

        if st.button("🧪 Run Rule Subset Ablation", type="primary"):
            with st.spinner("Executing rule subset ablation on benchmark circuit..."):
                framework = AblationFramework(
                    test_circuits=circuits_for_ablation,
                    gnn_model=gnn_model,
                    max_iterations=10,
                    max_evaluations_per_iter=10,
                )
                ablation_subsets = {
                    "Clifford Inverses Only": [SelfInverseCancellation(), CNOTCancellation(), IdentityRemoval()],
                    "Rotations Only": [RotationCombination(), ZeroRotationRemoval(), IdentityRemoval()],
                    "Full Rule Library": get_default_rules(),
                }
                ablation_results = framework.run_rule_subset_ablation(rule_subsets=ablation_subsets)

                ablation_rows = []
                for name, res in ablation_results.items():
                    df = res.summary_df
                    if not df.empty:
                        row = df.iloc[0].to_dict()
                        row["Subset"] = name
                        ablation_rows.append(row)

                ablation_df = pd.DataFrame(ablation_rows)
                st.session_state["ablation_summary_df"] = ablation_df

            st.success("Ablation study completed!")

        if "ablation_summary_df" in st.session_state and not st.session_state["ablation_summary_df"].empty:
            ab_df: pd.DataFrame = st.session_state["ablation_summary_df"]
            st.markdown("##### 📋 **Rule Subset Performance Breakdown**")
            st.dataframe(ab_df[["Subset", "avg_gate_reduction", "avg_cnot_reduction", "avg_evaluations", "avg_runtime_sec"]], use_container_width=True)
