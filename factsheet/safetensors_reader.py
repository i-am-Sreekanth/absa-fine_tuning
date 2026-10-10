"""Minimal, dependency-free .safetensors header reader.

The safetensors format starts with an 8-byte little-endian header length,
followed by that many bytes of JSON describing every tensor's dtype/shape/
byte-offsets. Reading just that header (no tensor data) is enough to count
parameters, so this works without `safetensors` or `torch` installed.
"""
from __future__ import annotations

import json
import struct


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def total_param_count(path: str) -> int:
    header = read_safetensors_header(path)
    total = 0
    for tensor in header.values():
        numel = 1
        for d in tensor["shape"]:
            numel *= d
        total += numel
    return total


def tensor_count(path: str) -> int:
    return len(read_safetensors_header(path))
