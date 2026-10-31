"""Load the unchanged research module without adding global search paths.

The research module uses a flat `from build_and_evaluate import ...`. Supply
that name only while importing it, then restore the previous module registry.
Importing that utility defines functions; its guarded training CLI never runs.
The private module names avoid collisions with future component modules.
"""
import importlib.util
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType

from app.core.config import REPOSITORY_ROOT


_lock = RLock()
_inference: ModuleType | None = None


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Component 3 research module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_research_inference() -> ModuleType:
    global _inference
    with _lock:
        if _inference is None:
            scripts = REPOSITORY_ROOT / "research" / "component3" / "scripts"
            baseline = _load_module("_smart_apparel_c3_baseline", scripts / "build_and_evaluate.py")
            previous = sys.modules.get("build_and_evaluate")
            sys.modules["build_and_evaluate"] = baseline
            try:
                _inference = _load_module("_smart_apparel_c3_inference", scripts / "model_inference.py")
            finally:
                if previous is None:
                    sys.modules.pop("build_and_evaluate", None)
                else:
                    sys.modules["build_and_evaluate"] = previous
        return _inference
