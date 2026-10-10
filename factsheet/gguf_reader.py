"""Minimal, dependency-free GGUF header/tensor-table reader.

Only reads the file header and tensor metadata (name, shape, dtype, offset)
to compute parameter counts. Never loads tensor weight data, so it works
without `gguf`, `torch`, or `numpy` and is fast even on multi-GB files.

Format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

_SCALAR_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_STRING_TYPE = 8
_ARRAY_TYPE = 9


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def _read_str(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8")


def _skip_value(f, vtype: int) -> None:
    if vtype == _STRING_TYPE:
        _read_str(f)
    elif vtype == _ARRAY_TYPE:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _skip_value(f, elem_type)
    else:
        f.read(_SCALAR_SIZES[vtype])


def read_gguf_tensor_table(path: str) -> list[TensorInfo]:
    """Return every tensor's name/shape/dtype/offset from a GGUF file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (bad magic {magic!r})")
        (_version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))

        for _ in range(kv_count):
            _read_str(f)  # key
            (vtype,) = struct.unpack("<I", f.read(4))
            _skip_value(f, vtype)

        tensors = []
        for _ in range(tensor_count):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append(TensorInfo(name=name, shape=dims, ggml_type=ggml_type, offset=offset))
        return tensors


def total_param_count(path: str) -> int:
    return sum(t.numel for t in read_gguf_tensor_table(path))
