"""Helpers for the eel side-note figures (skipped by build_figures.py).

- post(): import eel_agent_training.py of this post.
- hbars(): horizontal bars with mono value labels.
- SERIES_NAME: the subtitle prefix.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from svglib import INK, Diagram

POST = Path(__file__).resolve().parents[1]
SERIES_NAME = "Weight-Space Programming"


def post() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eel_agent_training", POST / "eel_agent_training.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eel_agent_training"] = module
    spec.loader.exec_module(module)
    return module


def hbars(
    d: Diagram,
    x: float,
    y: float,
    rows: list[tuple[str, float, str, str]],
    vmin: float,
    vmax: float,
    width: float = 520,
    label_w: float = 300,
    pitch: float = 26,
    value_x: float | None = None,
) -> float:
    """rows: (label, value, value text, colour). Bars start at vmin. value_x: fixed x of the value column."""
    for i, (label, value, text, col) in enumerate(rows):
        yy = y + i * pitch
        d.text(x + label_w - 10, yy + 13, label, 12.5, INK, anchor="end")
        bw = max(2.0, width * (value - vmin) / (vmax - vmin))
        d.rect(x + label_w, yy, bw, 18, fill=col, stroke=None, rx=3)
        tx = value_x if value_x is not None else x + label_w + bw + 8
        d.text(tx, yy + 13.5, text, 12.5, INK, mono=True)
    return y + len(rows) * pitch
