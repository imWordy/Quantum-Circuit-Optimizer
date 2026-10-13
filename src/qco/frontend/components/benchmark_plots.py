"""Interactive Plotly benchmark and ablation charts for students."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METHOD_COLOR_MAP = {
    "RandomOptimizer": "#EF4444",        # Red
    "GreedyOptimizer": "#F59E0B",        # Amber
    "MLPOptimizer": "#3B82F6",           # Blue
    "GNNGuidedOptimizer": "#10B981",     # Emerald green
}


def render_reduction_comparison_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing average gate and CNOT reduction across optimizers."""
    fig = go.Figure()

    methods = summary_df["method"].tolist()
    gate_red = summary_df["avg_gate_reduction"].tolist()
    cnot_red = summary_df["avg_cnot_reduction"].tolist()

    colors = [METHOD_COLOR_MAP.get(m, "#6366F1") for m in methods]

    fig.add_trace(
        go.Bar(
            x=methods,
            y=gate_red,
            name="Avg Gate Reduction",
            marker_color=colors,
            text=[f"{v:.2f}" for v in gate_red],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=methods,
            y=cnot_red,
            name="Avg CNOT Reduction",
            marker_color="#A855F7",
            text=[f"{v:.2f}" for v in cnot_red],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Optimization Performance (Average Reductions)</b>",
            font=dict(color="#F8FAFC", size=14),
        ),
        barmode="group",
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(font=dict(color="#F8FAFC")),
        xaxis=dict(
            tickfont=dict(color="#94A3B8"),
            gridcolor="#334155",
        ),
        yaxis=dict(
            title=dict(text="Average Gates Reduced", font=dict(color="#94A3B8")),
            tickfont=dict(color="#94A3B8"),
            gridcolor="#334155",
        ),
        height=340,
    )
    return fig


def render_efficiency_scatter_plot(raw_df: pd.DataFrame) -> go.Figure:
    """Scatter plot showing Search Efficiency: Evaluations vs Gate Reduction per Circuit."""
    fig = px.scatter(
        raw_df,
        x="candidate_evaluations",
        y="gate_reduction",
        color="method",
        color_discrete_map=METHOD_COLOR_MAP,
        hover_data=["circuit_id", "runtime", "iterations"],
        labels={
            "candidate_evaluations": "Candidate Evaluations (Search Cost)",
            "gate_reduction": "Gate Reduction (Optimization Benefit)",
            "method": "Optimizer",
        },
    )

    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color="#F8FAFC")))

    fig.update_layout(
        title=dict(
            text="<b>Search Efficiency Trade-Off (Benefit vs. Evaluations)</b>",
            font=dict(color="#F8FAFC", size=14),
        ),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(font=dict(color="#F8FAFC")),
        xaxis=dict(
            tickfont=dict(color="#94A3B8"),
            gridcolor="#334155",
        ),
        yaxis=dict(
            tickfont=dict(color="#94A3B8"),
            gridcolor="#334155",
        ),
        height=340,
    )
    return fig


def render_runtime_bar_plot(summary_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing average search runtime per circuit."""
    methods = summary_df["method"].tolist()
    runtimes = summary_df["avg_runtime_sec"].tolist()
    colors = [METHOD_COLOR_MAP.get(m, "#6366F1") for m in methods]

    fig = go.Figure(
        go.Bar(
            x=runtimes,
            y=methods,
            orientation="h",
            marker_color=colors,
            text=[f"{r:.4f} s" for r in runtimes],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Average Runtime per Circuit (Seconds)</b>",
            font=dict(color="#F8FAFC", size=14),
        ),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            title=dict(text="Seconds", font=dict(color="#94A3B8")),
            tickfont=dict(color="#94A3B8"),
            gridcolor="#334155",
        ),
        yaxis=dict(
            tickfont=dict(color="#94A3B8"),
            gridcolor="#334155",
        ),
        height=260,
    )
    return fig
