"""
벤치마크 실행 함수들
"""
from __future__ import annotations

import time
import multiprocessing as mp

from benchmark_workers import model_compress_worker, codec_compress_worker


def run_model_single(tasks: list[tuple[str, int, int]]) -> tuple[int, int, float]:
    """모델 기반 순차(싱글) 압축"""
    total_orig = 0
    total_comp = 0

    t0 = time.perf_counter()
    for tsk in tasks:
        o, c = model_compress_worker(tsk)
        total_orig += o
        total_comp += c
    t1 = time.perf_counter()

    return total_orig, total_comp, t1 - t0


def run_model_multi(tasks: list[tuple[str, int, int]], num_workers: int) -> tuple[int, int, float]:
    """모델 기반 멀티프로세스 압축"""
    total_orig = 0
    total_comp = 0

    workers = min(mp.cpu_count(), num_workers)
    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        for o, c in pool.imap_unordered(model_compress_worker, tasks, chunksize=4):
            total_orig += o
            total_comp += c
    t1 = time.perf_counter()

    return total_orig, total_comp, t1 - t0


def run_codec_single(tasks: list[tuple[str, int, int]], codec: str) -> tuple[int, int, float]:
    """단일 코덱 순차(싱글) 압축"""
    total_orig = 0
    total_comp = 0

    t0 = time.perf_counter()
    for file_path_str, offset, chunk_size in tasks:
        o, c = codec_compress_worker((codec, file_path_str, offset, chunk_size))
        total_orig += o
        total_comp += c
    t1 = time.perf_counter()

    return total_orig, total_comp, t1 - t0


def run_codec_multi(tasks: list[tuple[str, int, int]], codec: str, num_workers: int) -> tuple[int, int, float]:
    """단일 코덱 멀티프로세스 압축"""
    total_orig = 0
    total_comp = 0

    workers = min(mp.cpu_count(), num_workers)
    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        args_iter = (
            (codec, file_path_str, offset, chunk_size)
            for (file_path_str, offset, chunk_size) in tasks
        )
        for o, c in pool.imap_unordered(codec_compress_worker, args_iter, chunksize=4):
            total_orig += o
            total_comp += c
    t1 = time.perf_counter()

    return total_orig, total_comp, t1 - t0
