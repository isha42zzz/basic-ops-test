#!/usr/bin/env python3
"""
CSV 基准测试共用逻辑：
- 客户端构造原始测试数据（数组）
- 对数组做序列化以便放入加密请求
- 服务端解包后执行基础运算测试
"""

from __future__ import annotations

import base64
import io
import time
import zlib
from typing import Any

import numpy as np
from csv_secure_channel import seal_json, unseal_json


def build_workload(profile: str) -> dict[str, dict[str, Any]]:
    full = {
        "search": {"n": 1_000_000, "q": 100, "limit_sec": 5.0},
        "add": {"n": 10_000_000, "limit_sec": 5.0},
        "mul": {"n": 10_000_000, "limit_sec": 5.0},
        "compare": {"n": 10_000_000, "limit_sec": 5.0},
        "intersection": {
            "n_a": 100_000_000,
            "n_b": 100_000,
            "n_inter": 50_000,
            "limit_sec": 60.0,
        },
    }
    quick = {
        "search": {"n": 200_000, "q": 100, "limit_sec": 5.0},
        "add": {"n": 2_000_000, "limit_sec": 5.0},
        "mul": {"n": 2_000_000, "limit_sec": 5.0},
        "compare": {"n": 2_000_000, "limit_sec": 5.0},
        "intersection": {
            "n_a": 5_000_000,
            "n_b": 50_000,
            "n_inter": 25_000,
            "limit_sec": 60.0,
        },
    }
    if profile == "quick":
        return quick
    if profile == "full":
        return full
    raise ValueError(f"不支持的 profile: {profile}")


def _pack_array(arr: np.ndarray) -> dict[str, Any]:
    if arr.dtype != np.int64:
        raise ValueError(f"array dtype must be int64, got={arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"array ndim must be 1, got={arr.ndim}")

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    packed = zlib.compress(buf.getvalue(), level=1)
    return {
        "encoding": "npy+zlib+base64",
        "shape": [int(arr.size)],
        "dtype": "int64",
        "payload_b64": base64.b64encode(packed).decode("ascii"),
    }


def _unpack_array(item: dict[str, Any]) -> np.ndarray:
    if not isinstance(item, dict):
        raise ValueError("array payload must be object")
    if item.get("encoding") != "npy+zlib+base64":
        raise ValueError(f"unsupported encoding: {item.get('encoding')}")

    payload_b64 = str(item.get("payload_b64", ""))
    if not payload_b64:
        raise ValueError("array payload_b64 is empty")

    raw = base64.b64decode(payload_b64.encode("ascii"))
    decomp = zlib.decompress(raw)
    arr = np.load(io.BytesIO(decomp), allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError("decoded payload is not ndarray")
    if arr.dtype != np.int64 or arr.ndim != 1:
        raise ValueError(f"decoded array must be int64 1d, got={arr.dtype}/{arr.ndim}")
    return arr


def generate_client_vectors(profile: str, seed: int) -> dict[str, Any]:
    workload = build_workload(profile)
    rng = np.random.default_rng(seed)

    search_cfg = workload["search"]
    search_data = rng.integers(
        0,
        10 * search_cfg["n"],
        size=search_cfg["n"],
        dtype=np.int64,
    )
    search_queries = rng.choice(search_data, size=search_cfg["q"], replace=False)

    add_cfg = workload["add"]
    add_a = rng.integers(0, 1_000_000, size=add_cfg["n"], dtype=np.int64)
    add_b = rng.integers(0, 1_000_000, size=add_cfg["n"], dtype=np.int64)

    mul_cfg = workload["mul"]
    mul_a = rng.integers(0, 1_000_000, size=mul_cfg["n"], dtype=np.int64)
    mul_b = rng.integers(0, 1_000_000, size=mul_cfg["n"], dtype=np.int64)

    cmp_cfg = workload["compare"]
    cmp_a = rng.integers(0, 1_000_000, size=cmp_cfg["n"], dtype=np.int64)
    cmp_b = rng.integers(0, 1_000_000, size=cmp_cfg["n"], dtype=np.int64)

    inter_cfg = workload["intersection"]
    inter_a = np.arange(1, inter_cfg["n_a"] + 1, dtype=np.int64)
    rng.shuffle(inter_a)
    in_part = rng.choice(inter_a, size=inter_cfg["n_inter"], replace=False)
    out_part = rng.integers(
        inter_cfg["n_a"] + 1,
        inter_cfg["n_a"] + 1 + 10 * inter_cfg["n_b"],
        size=inter_cfg["n_b"] - inter_cfg["n_inter"],
        dtype=np.int64,
    )
    inter_b = np.concatenate([in_part, out_part])
    rng.shuffle(inter_b)

    return {
        "search": {
            "limit_sec": float(search_cfg["limit_sec"]),
            "data": _pack_array(search_data),
            "queries": _pack_array(search_queries),
        },
        "add": {
            "limit_sec": float(add_cfg["limit_sec"]),
            "a": _pack_array(add_a),
            "b": _pack_array(add_b),
        },
        "mul": {
            "limit_sec": float(mul_cfg["limit_sec"]),
            "a": _pack_array(mul_a),
            "b": _pack_array(mul_b),
        },
        "compare": {
            "limit_sec": float(cmp_cfg["limit_sec"]),
            "a": _pack_array(cmp_a),
            "b": _pack_array(cmp_b),
        },
        "intersection": {
            "limit_sec": float(inter_cfg["limit_sec"]),
            "expected_intersection": int(inter_cfg["n_inter"]),
            "a": _pack_array(inter_a),
            "b": _pack_array(inter_b),
        },
    }


def build_client_bench_payload(profile: str, seed: int, key: bytes) -> dict[str, Any]:
    vectors = generate_client_vectors(profile=profile, seed=seed)
    sealed_vectors = {
        name: seal_json(payload, key) for name, payload in vectors.items()
    }
    return {
        "profile": profile,
        "seed": int(seed),
        "sealed_vectors": sealed_vectors,
    }


def _count_intersection_asymmetric(
    a: np.ndarray, b: np.ndarray, chunk_size: int = 5_000_000
) -> int:
    b_unique = np.unique(b)
    if b_unique.size == 0 or a.size == 0:
        return 0

    inter_count = 0
    for start in range(0, a.size, chunk_size):
        chunk = a[start : start + chunk_size]
        idx = np.searchsorted(b_unique, chunk, side="left")
        valid = idx < b_unique.size
        if np.any(valid):
            inter_count += int(np.count_nonzero(b_unique[idx[valid]] == chunk[valid]))
    return inter_count


def _run_search(sealed_vector: str, key: bytes) -> dict[str, Any]:
    t0 = time.perf_counter()
    vector = unseal_json(sealed_vector, key)
    limit_sec = float(vector["limit_sec"])
    data = _unpack_array(vector["data"])
    queries = _unpack_array(vector["queries"])
    decrypt_data_elapsed = time.perf_counter() - t0

    n = int(data.size)
    q = int(queries.size)

    t1 = time.perf_counter()
    data = data.copy()
    data.sort()
    pos = np.searchsorted(data, queries)
    valid = pos < n
    pos_safe = np.minimum(pos, n - 1)
    found = valid & (data[pos_safe] == queries)
    found_count = int(found.sum())
    compute_elapsed = time.perf_counter() - t1
    elapsed = decrypt_data_elapsed + compute_elapsed

    return {
        "name": "(1) 密文查找",
        "elapsed_sec": elapsed,
        "decrypt_data_sec": decrypt_data_elapsed,
        # Keep legacy field for backward compatibility with old clients.
        "decrypt_sec": decrypt_data_elapsed,
        "compute_sec": compute_elapsed,
        "limit_sec": limit_sec,
        "ok": bool(found_count == q and elapsed <= limit_sec),
        "detail": f"n={n}, q={q}, found={found_count}",
    }


def _run_add(sealed_vector: str, key: bytes) -> dict[str, Any]:
    t0 = time.perf_counter()
    vector = unseal_json(sealed_vector, key)
    limit_sec = float(vector["limit_sec"])
    a = _unpack_array(vector["a"])
    b = _unpack_array(vector["b"])
    decrypt_data_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = a + b
    compute_elapsed = time.perf_counter() - t1
    elapsed = decrypt_data_elapsed + compute_elapsed
    return {
        "name": "(2) 密文加法",
        "elapsed_sec": elapsed,
        "decrypt_data_sec": decrypt_data_elapsed,
        # Keep legacy field for backward compatibility with old clients.
        "decrypt_sec": decrypt_data_elapsed,
        "compute_sec": compute_elapsed,
        "limit_sec": limit_sec,
        "ok": bool(elapsed <= limit_sec),
        "detail": f"n={a.size}",
    }


def _run_mul(sealed_vector: str, key: bytes) -> dict[str, Any]:
    t0 = time.perf_counter()
    vector = unseal_json(sealed_vector, key)
    limit_sec = float(vector["limit_sec"])
    a = _unpack_array(vector["a"])
    b = _unpack_array(vector["b"])
    decrypt_data_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = a * b
    compute_elapsed = time.perf_counter() - t1
    elapsed = decrypt_data_elapsed + compute_elapsed
    return {
        "name": "(3) 密文乘法",
        "elapsed_sec": elapsed,
        "decrypt_data_sec": decrypt_data_elapsed,
        # Keep legacy field for backward compatibility with old clients.
        "decrypt_sec": decrypt_data_elapsed,
        "compute_sec": compute_elapsed,
        "limit_sec": limit_sec,
        "ok": bool(elapsed <= limit_sec),
        "detail": f"n={a.size}",
    }


def _run_compare(sealed_vector: str, key: bytes) -> dict[str, Any]:
    t0 = time.perf_counter()
    vector = unseal_json(sealed_vector, key)
    limit_sec = float(vector["limit_sec"])
    a = _unpack_array(vector["a"])
    b = _unpack_array(vector["b"])
    decrypt_data_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = a > b
    compute_elapsed = time.perf_counter() - t1
    elapsed = decrypt_data_elapsed + compute_elapsed
    return {
        "name": "(4) 密文比较",
        "elapsed_sec": elapsed,
        "decrypt_data_sec": decrypt_data_elapsed,
        # Keep legacy field for backward compatibility with old clients.
        "decrypt_sec": decrypt_data_elapsed,
        "compute_sec": compute_elapsed,
        "limit_sec": limit_sec,
        "ok": bool(elapsed <= limit_sec),
        "detail": f"n={a.size}",
    }


def _run_intersection(sealed_vector: str, key: bytes) -> dict[str, Any]:
    t0 = time.perf_counter()
    vector = unseal_json(sealed_vector, key)
    expected = int(vector["expected_intersection"])
    limit_sec = float(vector["limit_sec"])
    a = _unpack_array(vector["a"])
    b = _unpack_array(vector["b"])
    decrypt_data_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    if a.size >= 10 * b.size:
        inter_count = _count_intersection_asymmetric(a, b)
    else:
        inter_count = int(np.intersect1d(a, b).size)
    compute_elapsed = time.perf_counter() - t1
    elapsed = decrypt_data_elapsed + compute_elapsed

    return {
        "name": "(5) 密文求交",
        "elapsed_sec": elapsed,
        "decrypt_data_sec": decrypt_data_elapsed,
        # Keep legacy field for backward compatibility with old clients.
        "decrypt_sec": decrypt_data_elapsed,
        "compute_sec": compute_elapsed,
        "limit_sec": limit_sec,
        "ok": bool(inter_count == expected and elapsed <= limit_sec),
        "detail": f"A={a.size}, B={b.size}, inter={inter_count}",
    }


def run_plaintext_bench_from_payload(payload: dict[str, Any], key: bytes) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be object")

    sealed_vectors = payload.get("sealed_vectors")
    if not isinstance(sealed_vectors, dict):
        raise ValueError("payload.sealed_vectors must be object")

    results = [
        _run_search(str(sealed_vectors["search"]), key),
        _run_add(str(sealed_vectors["add"]), key),
        _run_mul(str(sealed_vectors["mul"]), key),
        _run_compare(str(sealed_vectors["compare"]), key),
        _run_intersection(str(sealed_vectors["intersection"]), key),
    ]

    return {
        "profile": str(payload.get("profile", "unknown")),
        "seed": int(payload.get("seed", -1)),
        "results": results,
        "all_pass": all(item["ok"] for item in results),
    }
