#!/usr/bin/env python3
"""
明文性能测试（非隐私计算）
覆盖：查找、加法、乘法、比较、求交
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchResult:
    name: str
    elapsed_sec: float
    limit_sec: float
    ok: bool
    detail: str


class PlaintextBench:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _count_intersection_asymmetric(
        a: np.ndarray, b: np.ndarray, chunk_size: int = 5_000_000
    ) -> int:
        """
        适配 |A| >> |B| 的求交计数：
        - 仅对较小侧 B 去重并排序一次
        - 分块扫描 A，使用 searchsorted 做向量化成员测试
        """
        b_unique = np.unique(b)
        if b_unique.size == 0 or a.size == 0:
            return 0

        inter_count = 0
        for start in range(0, a.size, chunk_size):
            chunk = a[start : start + chunk_size]
            idx = np.searchsorted(b_unique, chunk, side="left")
            valid = idx < b_unique.size
            if np.any(valid):
                inter_count += int(
                    np.count_nonzero(b_unique[idx[valid]] == chunk[valid])
                )
        return inter_count

    def test_search(
        self, n: int = 1_000_000, q: int = 100, limit_sec: float = 5.0
    ) -> BenchResult:
        """
        场景(1): 5秒内完成任意100万条数据集中查找100条数据。
        使用排序 + 二分查找（searchsorted）。

        计时口径：
        - 不计数据构造（生成 data / queries）
        - 计入从数据处理开始到拿到结果（排序、查找、命中结果提取）
        """
        data = self.rng.integers(0, 10 * n, size=n, dtype=np.int64)
        queries = self.rng.choice(data, size=q, replace=False)

        t0 = time.perf_counter()
        data.sort()
        pos = np.searchsorted(data, queries)
        valid = pos < n
        pos_safe = np.minimum(pos, n - 1)
        found = valid & (data[pos_safe] == queries)
        found_count = int(found.sum())
        elapsed = time.perf_counter() - t0

        return BenchResult(
            name="(1) 明文查找",
            elapsed_sec=elapsed,
            limit_sec=limit_sec,
            ok=bool(found_count == q and elapsed <= limit_sec),
            detail=f"n={n}, q={q}, found={found_count}",
        )

    def test_add(self, n: int = 10_000_000, limit_sec: float = 5.0) -> BenchResult:
        """
        场景(2): 5秒内完成任意两个1千万级数据集执行加法。
        """
        a = self.rng.integers(0, 1_000_000, size=n, dtype=np.int64)
        b = self.rng.integers(0, 1_000_000, size=n, dtype=np.int64)

        t0 = time.perf_counter()
        _ = a + b
        elapsed = time.perf_counter() - t0

        return BenchResult(
            name="(2) 明文加法",
            elapsed_sec=elapsed,
            limit_sec=limit_sec,
            ok=bool(elapsed <= limit_sec),
            detail=f"n={n}",
        )

    def test_mul(self, n: int = 10_000_000, limit_sec: float = 5.0) -> BenchResult:
        """
        场景(3): 5秒内完成任意两个1千万级数据集执行乘法。
        """
        a = self.rng.integers(0, 1_000_000, size=n, dtype=np.int64)
        b = self.rng.integers(0, 1_000_000, size=n, dtype=np.int64)

        t0 = time.perf_counter()
        _ = a * b
        elapsed = time.perf_counter() - t0

        return BenchResult(
            name="(3) 明文乘法",
            elapsed_sec=elapsed,
            limit_sec=limit_sec,
            ok=bool(elapsed <= limit_sec),
            detail=f"n={n}",
        )

    def test_compare(self, n: int = 10_000_000, limit_sec: float = 5.0) -> BenchResult:
        """
        场景(4): 5秒内完成任意两个1千万级数据集执行比较。
        比较条件：a > b
        """
        a = self.rng.integers(0, 1_000_000, size=n, dtype=np.int64)
        b = self.rng.integers(0, 1_000_000, size=n, dtype=np.int64)

        t0 = time.perf_counter()
        _ = a > b
        elapsed = time.perf_counter() - t0

        return BenchResult(
            name="(4) 明文比较",
            elapsed_sec=elapsed,
            limit_sec=limit_sec,
            ok=bool(elapsed <= limit_sec),
            detail=f"n={n}",
        )

    def test_intersection(
        self,
        n_a: int = 100_000_000,
        n_b: int = 100_000,
        n_inter: int = 50_000,
        limit_sec: float = 60.0,
    ) -> BenchResult:
        """
        场景(5): 1分钟内完成 A(1亿) 与 B(10万) 的求交，交集5万。

        为了稳定可复现：
        - A 构造为唯一 int64 序列后打乱
        - B 前 n_inter 来自 A，后续来自不重叠区间
        - 当 A 远大于 B 时，使用不对称求交计数优化
        """
        if n_inter > n_b:
            raise ValueError("n_inter 不能大于 n_b")

        a = np.arange(1, n_a + 1, dtype=np.int64)
        self.rng.shuffle(a)

        in_part = self.rng.choice(a, size=n_inter, replace=False)
        out_part_size = n_b - n_inter
        out_part = self.rng.integers(
            n_a + 1, n_a + 1 + 10 * n_b, size=out_part_size, dtype=np.int64
        )
        b = np.concatenate([in_part, out_part])
        self.rng.shuffle(b)

        t0 = time.perf_counter()
        # 不对称规模优先：避免对 A 做全量排序/去重。
        # 如果规模接近，回退到 intersect1d。
        if n_a >= 10 * n_b:
            inter_count = self._count_intersection_asymmetric(a, b)
        else:
            inter_count = int(np.intersect1d(a, b).size)
        elapsed = time.perf_counter() - t0

        return BenchResult(
            name="(5) 明文求交",
            elapsed_sec=elapsed,
            limit_sec=limit_sec,
            ok=bool(inter_count == n_inter and elapsed <= limit_sec),
            detail=f"A={n_a}, B={n_b}, inter={inter_count}",
        )


def print_result(r: BenchResult) -> None:
    status = "PASS" if r.ok else "FAIL"
    print(
        f"{r.name}: {status} | {r.elapsed_sec:.4f}s / limit {r.limit_sec:.2f}s | {r.detail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="明文性能测试代码")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    bench = PlaintextBench(seed=args.seed)

    results = [
        bench.test_search(),
        bench.test_add(),
        bench.test_mul(),
        bench.test_compare(),
        bench.test_intersection(),
    ]

    print("明文性能测试结果：")
    for r in results:
        print_result(r)


if __name__ == "__main__":
    main()
