"""Small standard-library bridge used by the WASM parity test."""

import ctypes
import json
import math
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[2]
lib = ctypes.CDLL(str(root / "build-optimization-native/optimization/libfg_optimization.so"))
lib.fgo_create.argtypes = [ctypes.c_char_p]
lib.fgo_create.restype = ctypes.c_uint32
lib.fgo_snapshot.argtypes = [ctypes.c_uint32]
lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
lib.fgo_snapshot_size.argtypes = [ctypes.c_uint32]
lib.fgo_step.argtypes = [ctypes.c_uint32]
lib.fgo_destroy.argtypes = [ctypes.c_uint32]
lib.fgo_error.restype = ctypes.c_char_p
payload = json.load(sys.stdin)
handle = lib.fgo_create(json.dumps(payload["config"]).encode())
if not handle:
    raise RuntimeError(lib.fgo_error().decode())
try:
    for _ in range(payload["steps"]):
        if lib.fgo_step(handle) < 0:
            raise RuntimeError(lib.fgo_error().decode())
    print(
        json.dumps([
            v if math.isfinite(v) else None
            for v in lib.fgo_snapshot(handle)[: lib.fgo_snapshot_size(handle)]
        ])
    )
finally:
    lib.fgo_destroy(handle)
