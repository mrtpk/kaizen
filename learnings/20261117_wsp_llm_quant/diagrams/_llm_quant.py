"""Helpers for the LLM quantization figures (skipped by build_figures.py).

- post(): import llm_quant.py of this post.
- SERIES_NAME: the subtitle prefix.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

POST = Path(__file__).resolve().parents[1]
SERIES_NAME = "Weight-Space Programming"


def post() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llm_quant", POST / "llm_quant.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_quant"] = module
    spec.loader.exec_module(module)
    return module
