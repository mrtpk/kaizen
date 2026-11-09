"""Helpers for the eel weight-space figures (skipped by build_figures.py).

- post(): import eel_weight_space.py of this post (its numbers come from study_results.json).
- runs(): the study's evaluation results, by tag.
- hbars(): horizontal bars with mono value labels.
- SERIES_NAME: the subtitle prefix.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

from svglib import INK, Diagram

POST = Path(__file__).resolve().parents[1]
SERIES_NAME = "Weight-Space Programming"


def post() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eel_weight_space", POST / "eel_weight_space.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eel_weight_space"] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def data() -> dict:
    return post().load()


def hbars(
    d: Diagram,
    x: float,
    y: float,
    rows: list[tuple[str, float, str, str]],
    vmin: float,
    vmax: float,
    width: float = 520,
    label_w: float = 260,
    pitch: float = 26,
) -> float:
    """rows: (label, value, value text, colour). Bars start at vmin. Returns the y after the last bar."""
    for i, (label, value, text, col) in enumerate(rows):
        yy = y + i * pitch
        d.text(x + label_w - 10, yy + 13, label, 12.5, INK, anchor="end")
        bw = max(2.0, width * (value - vmin) / (vmax - vmin))
        d.rect(x + label_w, yy, bw, 18, fill=col, stroke=None, rx=3)
        d.text(x + label_w + bw + 8, yy + 13.5, text, 12.5, INK, mono=True)
    return y + len(rows) * pitch
