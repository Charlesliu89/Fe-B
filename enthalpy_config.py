"""Shared constants for enthalpy calculation/plotting."""

from pathlib import Path

DEFAULT_FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 20
FONT_COLOR = "black"
FONT_WEIGHT = "bold"

# Plotly export settings
PLOTLY_EXPORT = {
    "width": 1280,
    "height": 960,
    "scale": 2,
}

PLOTLY_BASE_FONT = {
    "family": DEFAULT_FONT_FAMILY,
    "color": FONT_COLOR,
}

PLOTLY_ELEMENT_FONT = {
    **PLOTLY_BASE_FONT,
    "size": FONT_SIZE,
}

COLORBAR_LABEL_CONFIG = {
    "text": r"$\Delta H_{\mathrm{mix}}$ (kJ/mol)",
    "plotly_text": "ΔH_mix (kJ/mol)",
    "rotation_deg": 0,
    "mat_axes_position": (0.5, 1.02),
    "plotly_position": (1.05, 1.0),
    "plotly_xanchor": "center",
    "plotly_yanchor": "bottom",
    "font_size": 20,
    "font_weight": "bold",
    "font_color": FONT_COLOR,
    "font_family": DEFAULT_FONT_FAMILY,
}

# Sampling steps
BINARY_STEP = 0.001  # 0.1%
TERNARY_STEP = 0.01  # 1%
QUATERNARY_STEP = 0.01  # default 1% for preview density
QUATERNARY_MIN_STEP = 0.01

# Data constants
OMEGA_SHEETS = ("U0", "U1", "U2", "U3")
DEFAULT_DATABASE_PATH = Path("Data/Element pair data base matrices.xlsx")

# Geometry: regular tetrahedron vertices (barycentric embedding)
TETRA_VERTICES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.5, 3**0.5 / 2.0, 0.0),
    (0.5, 3**0.5 / 6.0, (2.0 / 3.0) ** 0.5),
]
