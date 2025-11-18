"""Plotly helpers for enthalpy visualizations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go

from enthalpy_config import (
    COLORBAR_LABEL_CONFIG,
    PLOTLY_BASE_FONT,
    PLOTLY_ELEMENT_FONT,
    PLOTLY_EXPORT,
    TETRA_VERTICES,
)
from enthalpy_core import barycentric_to_cartesian


def add_plotly_colorbar_label(fig: "go.Figure") -> None:
    cfg = COLORBAR_LABEL_CONFIG
    text = cfg.get("plotly_text") or cfg["text"]
    if str(cfg["font_weight"]).lower() == "bold":
        text = f"<b>{text}</b>"
    fig.add_annotation(
        x=cfg["plotly_position"][0],
        y=cfg["plotly_position"][1],
        xref="paper",
        yref="paper",
        text=text,
        textangle=-cfg["rotation_deg"],
        showarrow=False,
        xanchor=cfg["plotly_xanchor"],
        yanchor=cfg["plotly_yanchor"],
        font={
            "family": cfg["font_family"],
            "size": cfg["font_size"],
            "color": cfg["font_color"],
        },
    )


def apply_plotly_base_style(fig: "go.Figure") -> None:
    fig.update_layout(font=PLOTLY_BASE_FONT)


def write_plotly_image(fig: "go.Figure", target: Path) -> None:
    fig.write_image(str(target), **PLOTLY_EXPORT)


def build_binary_figure(combo: Sequence[str], fractions: Sequence[float], enthalpies: Sequence[float]) -> "go.Figure":
    fig = go.Figure(
        go.Scatter(
            x=[f * 100 for f in fractions],
            y=enthalpies,
            mode="lines+markers",
            hovertemplate=(
                f"{combo[0]}=%{{x:.3f}}%\n"
                f"{combo[1]}=%{{customdata:.3f}}%\n"
                "ΔH=%{y:.5f} kJ/mol"
            ),
            customdata=[(1.0 - f) * 100 for f in fractions],
        )
    )
    fig.update_layout(
        title=dict(
            text=r"Binary $\Delta H_{\mathrm{mix}}$: " + "-".join(combo),
            font=PLOTLY_ELEMENT_FONT,
        ),
        xaxis=dict(title=dict(text=f"{combo[0]} atomic %", font=PLOTLY_ELEMENT_FONT)),
        yaxis=dict(title=dict(text=r"$\Delta H_{\mathrm{mix}}$ (kJ/mol)", font=PLOTLY_ELEMENT_FONT)),
        template="plotly_white",
    )
    apply_plotly_base_style(fig)
    return fig


def build_ternary_figure(
    combo: Sequence[str],
    a_vals: Sequence[float],
    b_vals: Sequence[float],
    c_vals: Sequence[float],
    enthalpies: Sequence[float],
) -> "go.Figure":
    fig = go.Figure(
        go.Scatterternary(
            a=a_vals,
            b=b_vals,
            c=c_vals,
            mode="markers",
            marker=dict(
                size=6,
                color=enthalpies,
                colorscale="Plasma",
                colorbar=dict(
                    thickness=15,
                    len=0.75,
                    title=dict(
                        text=COLORBAR_LABEL_CONFIG.get("plotly_text"),
                        font=PLOTLY_ELEMENT_FONT,
                        side="top",
                    ),
                    xpad=40,
                ),
            ),
            hovertemplate=(
                f"{combo[0]}=%{{a:.2f}}%<br>"
                f"{combo[1]}=%{{b:.2f}}%<br>"
                f"{combo[2]}=%{{c:.2f}}%<br>"
                "ΔH=%{marker.color:.5f} kJ/mol"
            ),
        )
    )
    fig.update_layout(
        title=dict(text=f"Ternary ΔH<sub>mix</sub>: {'-'.join(combo)}", font=PLOTLY_ELEMENT_FONT),
        ternary=dict(
            sum=100,
            aaxis=dict(title=dict(text=combo[0], font=PLOTLY_ELEMENT_FONT)),
            baxis=dict(title=dict(text=combo[1], font=PLOTLY_ELEMENT_FONT)),
            caxis=dict(title=dict(text=combo[2], font=PLOTLY_ELEMENT_FONT)),
        ),
        margin=dict(l=60, r=140, t=80, b=60),
        template="plotly_white",
    )
    apply_plotly_base_style(fig)
    return fig


def build_quaternary_figure(
    combo: Sequence[str],
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    z_vals: Sequence[float],
    enthalpies: Sequence[float],
    fractions: Sequence[Sequence[float]],
) -> "go.Figure":
    vertex_labels = []
    vertices = TETRA_VERTICES
    for idx, name in enumerate(combo):
        vertex_labels.append(
            go.Scatter3d(
                x=[vertices[idx][0]],
                y=[vertices[idx][1]],
                z=[vertices[idx][2]],
                mode="text",
                text=[f"<b>{name}</b>"],
                textfont={**PLOTLY_ELEMENT_FONT, "size": PLOTLY_ELEMENT_FONT["size"] + 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edge_traces = []
    for a_idx, b_idx in edge_pairs:
        edge_traces.append(
            go.Scatter3d(
                x=[vertices[a_idx][0], vertices[b_idx][0]],
                y=[vertices[a_idx][1], vertices[b_idx][1]],
                z=[vertices[a_idx][2], vertices[b_idx][2]],
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    hover_lines = []
    for frac in fractions:
        hover_lines.append("<br>".join(f"{elem}={value * 100:.2f}%" for elem, value in zip(combo, frac)))

    scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="markers",
        marker=dict(
            size=5,
            color=enthalpies,
            colorscale="Plasma",
            opacity=0.85,
            colorbar=dict(
                title=dict(text=COLORBAR_LABEL_CONFIG.get("plotly_text"), font=PLOTLY_ELEMENT_FONT),
                len=0.7,
                thickness=18,
                xpad=50,
            ),
        ),
        hovertemplate="%{text}<br>ΔH=%{marker.color:.5f} kJ/mol",
        text=hover_lines,
    )

    hull = go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=[v[2] for v in vertices],
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        opacity=0.15,
        color="lightgray",
        flatshading=True,
        hoverinfo="skip",
        showscale=False,
        name="composition hull",
    )

    fig = go.Figure([hull, scatter, *edge_traces, *vertex_labels])
    fig.update_layout(
        title=dict(text=f"Quaternary ΔH<sub>mix</sub>: {'-'.join(combo)}", font=PLOTLY_ELEMENT_FONT),
        scene=dict(
            xaxis=dict(title="X", showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="white"),
            yaxis=dict(title="Y", showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="white"),
            zaxis=dict(title="Z", showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="white"),
            aspectmode="cube",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=20, r=140, t=80, b=20),
        template="plotly_white",
    )
    apply_plotly_base_style(fig)
    return fig
