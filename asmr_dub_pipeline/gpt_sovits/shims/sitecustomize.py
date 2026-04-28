from __future__ import annotations

import importlib.util
import sys
import types


class _NoOpFileWriter:
    def add_summary(self, *args: object, **kwargs: object) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _NoOpSummaryWriter:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.log_dir = kwargs.get("log_dir") or (args[0] if args else None)
        self._file_writer = _NoOpFileWriter()

    def add_scalar(self, *args: object, **kwargs: object) -> None:
        pass

    def add_scalars(self, *args: object, **kwargs: object) -> None:
        pass

    def add_histogram(self, *args: object, **kwargs: object) -> None:
        pass

    def add_image(self, *args: object, **kwargs: object) -> None:
        pass

    def add_audio(self, *args: object, **kwargs: object) -> None:
        pass

    def add_text(self, *args: object, **kwargs: object) -> None:
        pass

    def add_graph(self, *args: object, **kwargs: object) -> None:
        pass

    def add_embedding(self, *args: object, **kwargs: object) -> None:
        pass

    def _get_file_writer(self) -> _NoOpFileWriter:
        return self._file_writer

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


if importlib.util.find_spec("tensorboard") is None:
    tensorboard_module = types.ModuleType("torch.utils.tensorboard")

    tensorboard_module.SummaryWriter = _NoOpSummaryWriter
    tensorboard_module.FileWriter = _NoOpSummaryWriter
    tensorboard_module.RecordWriter = object
    sys.modules.setdefault("torch.utils.tensorboard", tensorboard_module)
