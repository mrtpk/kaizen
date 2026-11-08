"""Helpers for the LiDAR Road Detection figures (skipped by build_figures.py).

- script(): import a script of the series, so a figure uses the script's numbers.
- heatmap(): draw a 400 x 200 grid image as blocks of cells.
- hbars(): horizontal bars with mono value labels.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from svglib import INK, LINE, MUTE, Diagram

LEARNINGS = Path(__file__).resolve().parents[2]


def script(folder: str, name: str) -> ModuleType:
    """Import learnings/<folder>/<name>.py without running its main()."""
    path = LEARNINGS / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _mix(c0: str, c1: str, t: float) -> str:
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def heatmap(d: Diagram, x: float, y: float, img: np.ndarray, title: str, col: str,
            block: int = 4, px: float = 2.0, lo: float | None = None, hi: float | None = None,
            note: str = "", mask: np.ndarray | None = None) -> float:
    """Draw img (400 x 200, row 0 = far) as blocks of block x block cells.

    mask marks the occupied cells (default: img != 0).
    Each block is px * block pixels wide. Empty blocks stay light gray; the value of
    an occupied block is the mean over its occupied cells, mapped from a light tint
    (lo) to the strong colour (hi). Returns the width of the drawing.
    """
    rows, cols = img.shape
    br, bc = rows // block, cols // block
    occ = (img != 0) if mask is None else mask
    s = np.where(occ, img, 0.0).reshape(br, block, bc, block).sum(axis=(1, 3))
    n = occ.reshape(br, block, bc, block).sum(axis=(1, 3))
    val = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    finite = val[np.isfinite(val)]
    lo = float(finite.min()) if lo is None else lo
    hi = float(finite.max()) if hi is None else hi
    size = block * px
    w, h = bc * size, br * size
    d.text(x + w / 2, y - 10, title, 13, INK, "bold", "middle")
    d.rect(x, y, w, h, fill="#f3f4f6", stroke=LINE, rx=0, sw=1)
    for i in range(br):
        for j in range(bc):
            v = val[i, j]
            if np.isfinite(v):
                t = 0.15 + 0.85 * (0.0 if hi == lo else (v - lo) / (hi - lo))
                d.add(f'<rect x="{x + j * size:.1f}" y="{y + i * size:.1f}" width="{size:.1f}" '
                      f'height="{size:.1f}" fill="{_mix("#ffffff", col, min(max(t, 0.0), 1.0))}"/>')
    if note:
        d.text(x + w / 2, y + h + 18, note, 12, MUTE, anchor="middle", mono=True)
    return w


def hbars(d: Diagram, x: float, y: float, rows: list[tuple[str, float, str, str]],
          vmax: float, width: float = 520, label_w: float = 170, pitch: float = 26) -> float:
    """rows: (label, value, value text, colour). Returns the y after the last bar."""
    for i, (label, value, text, col) in enumerate(rows):
        yy = y + i * pitch
        d.text(x + label_w - 10, yy + 13, label, 12.5, INK, anchor="end")
        bw = max(2.0, width * value / vmax)
        d.rect(x + label_w, yy, bw, 18, fill=col, stroke=None, rx=3)
        d.text(x + label_w + bw + 8, yy + 13.5, text, 12.5, INK, mono=True)
    return y + len(rows) * pitch
