"""llama.cpp builds shipped in the image.

Each engine is a directory under the engines dir with bin/llama-server,
bin/llama-bench and an engine.json written at image build time:

    {"name": "kmic", "label": "...", "repo": "...", "commit": "...",
     "cuda_archs": "60;61;70", "multi_gpu_split": "tensor",
     "env": {"GGML_CUDA_P2P": "1"}, "notes": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path


# A server that is already running somewhere else (another engine such as
# Strata, or a llama-server you started yourself). The kit only sends it
# requests: speed and offload need llama.cpp's own tools and are not offered.
EXTERNAL = {"name": "external", "label": "External server (URL)", "external": True, "available": True,
            "repo": "", "commit": "", "cuda_archs": "", "multi_gpu_split": "layer", "env": {},
            "notes": "Benchmarks an OpenAI-compatible server that is already running (server and agent "
                     "modules only). Results are marked not comparable."}


def discover(engines_dir: Path) -> list[dict]:
    out = []
    for d in sorted(Path(engines_dir).glob("*/engine.json")):
        try:
            e = json.loads(d.read_text())
        except ValueError:
            continue
        root = d.parent
        e["bin_dir"] = str(root / "bin")
        e["available"] = all((root / "bin" / b).exists() for b in ("llama-server", "llama-bench"))
        e.setdefault("env", {})
        e.setdefault("multi_gpu_split", "layer")
        out.append(e)
    return out + [dict(EXTERNAL)]


SPLIT_MODES = ("layer", "tensor")


def split_args(engine: dict, gpus: list[int], override: str | None = None) -> list[str]:
    """Split mode (the run's choice, else the engine default) and an even split."""
    if len(gpus) <= 1:
        return ["-sm", "none"]
    mode = override if override in SPLIT_MODES else engine.get("multi_gpu_split", "layer")
    return ["-sm", mode, "-ts", "/".join("1" for _ in gpus)]


def ngl_args(spec: dict) -> list[str]:
    """GPU layers for a llama.cpp launch: the run's -ngl, or 99 (all
    layers) when the spec leaves it unset."""
    n = spec.get("ngl")
    return ["-ngl", "99" if n is None else str(n)]


def env_for(engine: dict, gpus: list[int]) -> dict:
    return {**engine.get("env", {}), "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)}
