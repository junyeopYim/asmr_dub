from __future__ import annotations

import gc
from importlib import import_module
from typing import Any


def clear_gpu_vram(stage: str | None = None) -> None:
    """Release process-local GPU cache after a heavyweight pipeline stage.

    The pipeline keeps PyTorch optional for lightweight installs, so this helper
    imports it lazily and treats CUDA cleanup as best-effort.
    """
    _ = stage
    gc.collect()
    try:
        torch: Any = import_module("torch")
    except Exception:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if not bool(cuda.is_available()):
            return
        cuda.empty_cache()
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        return
    gc.collect()
