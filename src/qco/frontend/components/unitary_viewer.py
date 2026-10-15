"""Interactive Unitary Matrix Heatmap Visualizer for QCO.

Renders Plotly 2D heatmap plots of complex unitary matrices (Real, Imaginary,
Magnitude, Phase) and unitary product distance comparisons.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_unitary_heatmap(
    unitary: np.ndarray,
    title: str = "Unitary Matrix",
    mode: str = "real_imag",
    max_dim_display: int = 32,
) -> go.Figure:
    """Create an interactive Plotly heatmap visualization of a unitary matrix.

    Args:
        unitary: Complex numpy ndarray of shape (2^n, 2^n).
        title: Plot title.
        mode: Display mode - 'real_imag' (side-by-side Real and Imaginary)
              or 'magnitude_phase' (side-by-side Magnitude and Phase).
        max_dim_display: Threshold above which downsampling/warning is advised.

    Returns:
        Plotly Figure object.
    """
    dim = unitary.shape[0]
    display_matrix = unitary[:max_dim_display, :max_dim_display]
    ticks = [f"|{bin(i)[2:].zfill(int(np.log2(dim)))}⟩" if dim <= 16 else str(i) for i in range(display_matrix.shape[0])]

    if mode == "magnitude_phase":
        mag = np.abs(display_matrix)
        phase = np.angle(display_matrix)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"Magnitude |U| (shape {dim}x{dim})", "Phase Angle arg(U) [rad]"),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Heatmap(
                z=mag,
                x=ticks,
                y=ticks,
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title=dict(text="Mag")),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=phase,
                x=ticks,
                y=ticks,
                colorscale="Twilight",
                zmin=-np.pi,
                zmax=np.pi,
                colorbar=dict(title=dict(text="Rad")),
            ),
            row=1,
            col=2,
        )
    else:  # default 'real_imag'
        real_part = np.real(display_matrix)
        imag_part = np.imag(display_matrix)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"Real Part Re(U) ({dim}x{dim})", f"Imaginary Part Im(U) ({dim}x{dim})"),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Heatmap(
                z=real_part,
                x=ticks,
                y=ticks,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title=dict(text="Re")),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=imag_part,
                x=ticks,
                y=ticks,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title=dict(text="Im")),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" + (f" (Truncated to first {max_dim_display} rows)" if dim > max_dim_display else ""),
            font=dict(size=15, color="#F8FAFC"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.7)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        font=dict(color="#94A3B8"),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Invert y-axis to match standard matrix indexing top-to-bottom
    fig.update_yaxes(autorange="reversed")

    return fig


def create_product_distance_heatmap(
    product_matrix: np.ndarray,
    expected_matrix: np.ndarray,
    title: str = "Verification Residual: |U_A U_B† - e^(iφ) I|",
) -> go.Figure:
    """Create heatmap of the residual error matrix |U_A U_B† - e^(iφ) I|.

    In a valid equivalent rewrite, all elements in this residual matrix are <= tolerance (approx 0).
    Any non-zero element highlights an equivalence breakdown.
    """
    residual = np.abs(product_matrix - expected_matrix)
    dim = residual.shape[0]
    ticks = [f"|{bin(i)[2:].zfill(int(np.log2(dim)))}⟩" if dim <= 16 else str(i) for i in range(dim)]

    fig = go.Figure(
        data=go.Heatmap(
            z=residual,
            x=ticks,
            y=ticks,
            colorscale="Reds",
            zmin=0.0,
            colorbar=dict(title=dict(text="|Residual|")),
        )
    )

    max_err = float(np.max(residual))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b> (Max Residual: <code>{max_err:.2e}</code>)",
            font=dict(size=14, color="#F8FAFC"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.7)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        font=dict(color="#94A3B8"),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    return fig
