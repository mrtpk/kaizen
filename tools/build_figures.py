"""Build the figures of one post: run each diagram script, then convert SVG to PNG.

Usage (from the kaizen root):
    uv run --with cairosvg python tools/build_figures.py learnings/<post folder>

For each `<post>/diagrams/<name>.py` (files that start with "_" are helpers and are skipped):
  1. run it to write `<post>/resources/<name>.svg`
  2. convert the SVG to `<post>/resources/<name>.png` at 2x scale

The diagram scripts import `svglib` from `tools/diagrams/`. Rules: STYLE.md section "Figures".
"""

import os
import subprocess
import sys
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "tools" / "diagrams"
SCALE = 2


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    post = Path(sys.argv[1]).resolve()
    scripts = sorted(p for p in (post / "diagrams").glob("*.py") if not p.name.startswith("_"))
    if not scripts:
        print(f"no diagram scripts in {post / 'diagrams'}")
        return 1
    out_dir = post / "resources"
    out_dir.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(LIB), str(post / "diagrams")]))
    failed = 0
    for script in scripts:
        svg = out_dir / f"{script.stem}.svg"
        png = out_dir / f"{script.stem}.png"
        proc = subprocess.run([sys.executable, str(script), str(svg)], capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            failed += 1
            print(f"FAIL {script.name}\n{proc.stderr}")
            continue
        if proc.stderr.strip():
            print(f"WARN {script.name}: {proc.stderr.strip()}")
        cairosvg.svg2png(url=str(svg), write_to=str(png), scale=SCALE)
        print(f"ok   {png.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
