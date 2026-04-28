from __future__ import annotations

from pathlib import Path


class _ILoc:
    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, key: tuple[int, int]) -> str:
        row_index, column_index = key
        return self._frame._rows[row_index][column_index]


class DataFrame:
    def __init__(self, rows: list[list[str]]) -> None:
        self._rows = rows
        self.iloc = _ILoc(self)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: slice) -> DataFrame:
        if isinstance(key, slice):
            return DataFrame(self._rows[key])
        raise TypeError(f"unsupported pandas shim key: {key!r}")

    def __repr__(self) -> str:
        return f"<pandas.DataFrame rows={len(self._rows)}>"


def read_csv(path: str | Path, *args: object, **kwargs: object) -> DataFrame:
    delimiter = str(kwargs.get("delimiter") or kwargs.get("sep") or ",")
    encoding = str(kwargs.get("encoding") or "utf-8")
    header = kwargs.get("header", "infer")
    rows: list[list[str]] = []
    with open(path, encoding=encoding) as handle:
        for line in handle:
            stripped = line.rstrip("\n\r")
            if stripped:
                rows.append(stripped.split(delimiter))
    if (header == "infer" or header == 0) and rows:
        rows = rows[1:]
    elif isinstance(header, int) and header > 0:
        rows = rows[header + 1 :]
    return DataFrame(rows)


__all__ = ["DataFrame", "read_csv"]
