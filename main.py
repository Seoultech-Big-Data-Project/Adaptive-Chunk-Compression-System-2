#!/usr/bin/env python
"""
benchmark_all.py

- raw/ 디렉토리의 파일들을 16MB 청크로 나눔
- 모델 기반 (model_sample_128KB.json) 싱글/멀티 벤치마크
- 단일 코덱(zstd, lz4, snappy) 싱글/멀티 벤치마크

결과를 순차적으로 출력
"""

from __future__ import annotations

import time
from pathlib import Path
import multiprocessing as mp

import numpy as np
import zstandard as zstd
import lz4.frame
import snappy
import xgboost as xgb

from features import (
    sample_bytes,
    extract_all_features,
    FEATURE_KEYS,
    FEATURE_COLUMNS,
)

# =========================
# 공통 설정
# =========================

RAW_DIR = Path("raw")
MODEL_PATH = Path("model_sample_128KB.json")

CHUNK_SIZE_BYTES = 16 * 1024 * 1024   # 16MB
BENCH_SAMPLE_SIZE = 128 * 1024        # 128KB 샘플링

# 너무 오래 걸리면 앞에서부터 일부 청크만 써보고 싶을 때 (None이면 전체)
MAX_CHUNKS = None  # 예: 200

# # 병렬 프로세스 개수 (None이면 cpu_count())
# NUM_WORKERS = None

# 모델용 코덱 라벨 매핑
CODECS = ["zstd", "lz4", "snappy"]
CODEC_TO_LABEL = {c: i for i, c in enumerate(CODECS)}
LABEL_TO_CODEC = {i: c for i, c in enumerate(CODECS)}

# zstd용 공용 compressor (프로세스별 lazy init)
_ZSTD = None


# =========================
# 청크 목록 생성
# =========================

def build_tasks() -> list[tuple[str, int, int]]:
    """
    raw/ 디렉토리의 파일들을 16MB 청크로 쪼갠 작업 리스트 생성.
    각 항목: (파일경로 문자열, offset, chunk_size)
    """
    tasks: list[tuple[str, int, int]] = []

    for file_path in RAW_DIR.iterdir():
        if not file_path.is_file():
            continue
        size = file_path.stat().st_size
        offset = 0
        while offset < size:
            tasks.append((str(file_path), offset, CHUNK_SIZE_BYTES))
            offset += CHUNK_SIZE_BYTES

    if MAX_CHUNKS is not None:
        tasks = tasks[:MAX_CHUNKS]

    return tasks


# =========================
# 1. 모델 기반 워커 & 실행 함수
# =========================

_MODEL: xgb.Booster | None = None  # 프로세스별 lazy init


def model_compress_worker(task: tuple[str, int, int]) -> tuple[int, int]:
    """
    단일 청크에 대해:
      - 파일에서 읽기
      - 샘플링 + 피처 추출
      - 모델로 코덱 선택
      - 선택된 코덱으로 압축
    반환: (원본 크기, 압축 후 크기)
    """
    import lz4.frame
    import snappy

    global _MODEL, _ZSTD

    file_path_str, offset, chunk_size = task
    file_path = Path(file_path_str)

    with file_path.open("rb") as f:
        f.seek(offset)
        data = f.read(chunk_size)

    orig_size = len(data)
    if orig_size == 0:
        return 0, 0

    # 모델 & zstd compressor lazy init
    if _MODEL is None:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        _MODEL = booster
    if _ZSTD is None:
        _ZSTD = zstd.ZstdCompressor()

    # 1) 샘플링 + 피처
    sampled = sample_bytes(data, max_len=BENCH_SAMPLE_SIZE)
    feats = extract_all_features(sampled)
    feat_vec = np.array(
        [[feats[k] for k in FEATURE_KEYS] + [float(orig_size)]],
        dtype=np.float32,
    )
    drow = xgb.DMatrix(feat_vec, feature_names=FEATURE_COLUMNS)

    # 2) 모델 예측
    prob = _MODEL.predict(drow)[0]
    pred_label = int(prob.argmax())
    pred_codec = LABEL_TO_CODEC.get(pred_label, "lz4")  # 방어적으로 fallback

    # 3) 압축
    if pred_codec == "zstd":
        comp = _ZSTD.compress(data)
    elif pred_codec == "lz4":
        comp = lz4.frame.compress(data)
    else:  # "snappy"
        comp = snappy.compress(data)

    comp_size = len(comp)
    return orig_size, comp_size


def run_model_single(tasks: list[tuple[str, int, int]]):
    """모델 기반 순차(싱글) 압축."""
    total_orig = 0
    total_comp = 0

    t0 = time.perf_counter()
    for tsk in tasks:
        o, c = model_compress_worker(tsk)
        total_orig += o
        total_comp += c
    t1 = time.perf_counter()

    elapsed = t1 - t0
    return total_orig, total_comp, elapsed


def run_model_multi(tasks: list[tuple[str, int, int]], num_workers: int):
    """모델 기반 멀티프로세스 압축."""
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

    elapsed = t1 - t0
    return total_orig, total_comp, elapsed


# =========================
# 2. 단일 코덱 워커 & 실행 함수
# =========================

def codec_compress_worker(args: tuple[str, str, int, int]) -> tuple[int, int]:
    """
    단일 청크에 대해:
      - 파일에서 읽기
      - 주어진 코덱으로 압축
    args: (codec, file_path_str, offset, chunk_size)
    반환: (원본 크기, 압축 후 크기)
    """
    import lz4.frame
    import snappy

    global _ZSTD

    codec, file_path_str, offset, chunk_size = args
    file_path = Path(file_path_str)

    with file_path.open("rb") as f:
        f.seek(offset)
        data = f.read(chunk_size)

    orig_size = len(data)
    if orig_size == 0:
        return 0, 0

    if codec == "zstd":
        if _ZSTD is None:
            _ZSTD = zstd.ZstdCompressor()
        comp = _ZSTD.compress(data)
    elif codec == "lz4":
        comp = lz4.frame.compress(data)
    elif codec == "snappy":
        comp = snappy.compress(data)
    else:
        raise ValueError(f"지원하지 않는 코덱: {codec}")

    comp_size = len(comp)
    return orig_size, comp_size


def run_codec_single(tasks: list[tuple[str, int, int]], codec: str):
    """단일 코덱 순차(싱글) 압축."""
    total_orig = 0
    total_comp = 0

    t0 = time.perf_counter()
    for file_path_str, offset, chunk_size in tasks:
        o, c = codec_compress_worker((codec, file_path_str, offset, chunk_size))
        total_orig += o
        total_comp += c
    t1 = time.perf_counter()

    elapsed = t1 - t0
    return total_orig, total_comp, elapsed


def run_codec_multi(tasks: list[tuple[str, int, int]], codec: str, num_workers: int):
    """단일 코덱 멀티프로세스 압축."""
    total_orig = 0
    total_comp = 0

    workers = min(mp.cpu_count(), num_workers)
    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        # codec을 task마다 같이 넘겨줌
        args_iter = (
            (codec, file_path_str, offset, chunk_size)
            for (file_path_str, offset, chunk_size) in tasks
        )
        for o, c in pool.imap_unordered(codec_compress_worker, args_iter, chunksize=4):
            total_orig += o
            total_comp += c
    t1 = time.perf_counter()

    elapsed = t1 - t0
    return total_orig, total_comp, elapsed


# =========================
# 공통 출력 유틸
# =========================

def print_result_block(
    title: str,
    single_res: tuple[int, int, float],
    multi_res: tuple[int, int, float],
):
    s_orig, s_comp, s_time = single_res
    m_orig, m_comp, m_time = multi_res

    s_mb = s_orig / (1024 * 1024)
    m_mb = m_orig / (1024 * 1024)

    s_ratio = s_comp / s_orig if s_orig > 0 else float("inf")
    m_ratio = m_comp / m_orig if m_orig > 0 else float("inf")

    s_throughput = s_mb / s_time if s_time > 0 else 0.0
    m_throughput = m_mb / m_time if m_time > 0 else 0.0

    speedup = s_time / m_time if (s_time > 0 and m_time > 0) else float("inf")

    print(f"\n===== {title} =====")
    print("[싱글]")
    print(f"  - 총 원본: {s_mb:.2f} MB")
    print(f"  - 총 압축: {s_comp / (1024*1024):.2f} MB")
    print(f"  - 압축률: {s_ratio:.4f}")
    print(f"  - 소요 시간: {s_time:.2f} 초")
    print(f"  - 처리 속도: {s_throughput:.2f} MB/s")

    print("\n[멀티]")
    print(f"  - 총 원본: {m_mb:.2f} MB")
    print(f"  - 총 압축: {m_comp / (1024*1024):.2f} MB")
    print(f"  - 압축률: {m_ratio:.4f}")
    print(f"  - 소요 시간: {m_time:.2f} 초")
    print(f"  - 처리 속도: {m_throughput:.2f} MB/s")

    print(f"\n[비교] 속도 배율 (single / multi): {speedup:.2f}x")


# =========================
# 메인
# =========================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    raw_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if not raw_files:
        raise RuntimeError(f"raw 디렉토리에 파일이 없습니다: {RAW_DIR}")
    
    print(f"[INFO] 모델: {MODEL_PATH}")
    print(f"[INFO] raw 디렉토리: {RAW_DIR} (파일 {len(raw_files)}개)")
    for p in raw_files:
        print(f"  - {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"\n[INFO] 청크 크기: {CHUNK_SIZE_BYTES // (1024*1024)} MB")
    print(f"[INFO] 샘플링 크기: {BENCH_SAMPLE_SIZE // 1024} KB")
    tasks = build_tasks()
    print(f"[INFO] 대상 청크 수: {len(tasks)}")

    if MAX_CHUNKS is not None:
        print(f"[INFO] (MAX_CHUNKS = {MAX_CHUNKS} → 앞 {MAX_CHUNKS}개만 사용)")

    NUM_WORKER_LIST = [4, 8, 16, 32, 64, 128]

    for num_workers in NUM_WORKER_LIST:
        print(f"\n===== NUM_WORKERS = {num_workers} =====") 

        # ---------- 1. 모델 기반 ----------
        print("\n>>> 1) 모델 기반 싱글/멀티 벤치마크")
        model_single = run_model_single(tasks)
        model_multi = run_model_multi(tasks, num_workers)
        print_result_block("모델 기반 (adaptive)", model_single, model_multi)

        # ---------- 2. 단일 코덱들 ----------
        for codec in ["zstd", "lz4", "snappy"]:
            print(f"\n>>> 2) 단일 코덱 {codec} 싱글/멀티 벤치마크")
            codec_single = run_codec_single(tasks, codec)
            codec_multi = run_codec_multi(tasks, codec, num_workers)
            print_result_block(f"단일 코덱: {codec}", codec_single, codec_multi)


if __name__ == "__main__":
    main()
