#!/usr/bin/env python
"""
benchmark_overhead_of_model.py

모델 기반 압축에서만 멀티프로세싱 오버헤드를 분석하는 코드.

- 파일 읽기 시간은 제외
- 청크별로 다음 3가지만 측정:
    1) 피처 추출 시간 (feature extraction)
    2) 모델 예측 시간 (XGBoost predict)
    3) 실제 압축 시간 (선택된 코덱으로 full chunk 압축)

- NUM_WORKER_LIST 에 담긴 워커 수에 대해 각각 실행하고,
  워커 수별로 위 3가지가 전체 시간에서 차지하는 비율을 계산.

- 결과는 콘솔에 출력하고,
  워커 수(x축) vs 시간 비율(y축)의 스택 바 그래프를 저장.
"""

from __future__ import annotations

import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import xgboost as xgb
import zstandard as zstd
import lz4.frame
import snappy
import matplotlib.pyplot as plt

# =========================
# 1. 경로 및 설정 import
# =========================

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ 기존 benchmark.py 와 동일한 스타일로 import
from benchmark_config import (  # type: ignore
    RAW_DIR,
    MODEL_PATH,
    CHUNK_SIZE_BYTES,
    BENCH_SAMPLE_SIZE,
    MAX_CHUNKS,
    NUM_WORKER_LIST,
    LABEL_TO_CODEC,
    GRAPH_PATH,
)
from benchmark_utils import build_tasks  # type: ignore
from features.features import (  # type: ignore
    sample_bytes,
    extract_all_features,
    FEATURE_KEYS,
    FEATURE_COLUMNS,
)

# =========================
# 2. 전역 모델 / 코덱 (프로세스별 lazy init)
# =========================

_MODEL: xgb.Booster | None = None
_ZSTD = None  # zstd.ZstdCompressor 인스턴스 (lazy)


def _load_model() -> xgb.Booster:
    """프로세스별 Booster lazy 로드"""
    global _MODEL
    if _MODEL is None:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        _MODEL = booster
    return _MODEL


def _compress_with_codec(codec: str, data: bytes) -> bytes:
    """선택된 코덱으로 실제 압축 수행 (압축 시간 측정용)"""
    global _ZSTD
    if codec == "zstd":
        if _ZSTD is None:
            _ZSTD = zstd.ZstdCompressor()
        return _ZSTD.compress(data)
    elif codec == "lz4":
        return lz4.frame.compress(data)
    elif codec == "snappy":
        return snappy.compress(data)
    else:
        raise ValueError(f"지원하지 않는 코덱: {codec}")


# =========================
# 3. 워커 함수 (파일 읽기 시간 제외)
# =========================

def model_overhead_worker(task: tuple[str, int, int]) -> tuple[float, float, float]:
    """
    단일 청크에 대해:
    - 파일 읽기 시간은 제외
    - 피처 추출, 모델 예측, 실제 압축 시간만 측정

    Args:
        task: (file_path_str, offset, chunk_size)

    Returns:
        (feat_time, pred_time, comp_time)
    """
    file_path_str, offset, chunk_size = task
    file_path = Path(file_path_str)

    # --- 파일 읽기 (시간 측정 ❌) ---
    with file_path.open("rb") as f:
        f.seek(offset)
        data = f.read(chunk_size)

    orig_size = len(data)
    if orig_size == 0:
        return 0.0, 0.0, 0.0

    # --- (1) 피처 추출 시간 ---
    t_feat0 = time.perf_counter()
    sampled = sample_bytes(data, max_len=BENCH_SAMPLE_SIZE)
    feats = extract_all_features(sampled)
    feat_vec = np.array(
        [[feats[k] for k in FEATURE_KEYS] + [float(orig_size)]],
        dtype=np.float32,
    )
    t_feat1 = time.perf_counter()
    feat_time = t_feat1 - t_feat0

    # --- (2) 모델 예측 시간 ---
    booster = _load_model()
    t_pred0 = time.perf_counter()
    drow = xgb.DMatrix(feat_vec, feature_names=FEATURE_COLUMNS)
    prob = booster.predict(drow)[0]
    t_pred1 = time.perf_counter()
    pred_time = t_pred1 - t_pred0

    if prob.ndim == 0:
        pred_label = int(prob > 0.5)
    else:
        pred_label = int(prob.argmax())
    pred_codec = LABEL_TO_CODEC.get(pred_label, "lz4")

    # --- (3) 실제 압축 시간 ---
    t_comp0 = time.perf_counter()
    _ = _compress_with_codec(pred_codec, data)
    t_comp1 = time.perf_counter()
    comp_time = t_comp1 - t_comp0

    return feat_time, pred_time, comp_time


# =========================
# 4. 메인 루프
# =========================

def main():
    print("=" * 60)
    print("모델 기반 압축 멀티프로세싱 오버헤드 분석 (파일 I/O 제외)")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR가 존재하지 않습니다: {RAW_DIR}")

    raw_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if not raw_files:
        raise RuntimeError(f"raw 디렉토리에 파일이 없습니다: {RAW_DIR}")

    print(f"[INFO] raw 디렉토리: {RAW_DIR} (파일 {len(raw_files)}개)")
    for p in raw_files:
        print(f"  - {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"\n[INFO] 청크 크기: {CHUNK_SIZE_BYTES // (1024*1024)} MB")
    print(f"[INFO] 샘플링 크기: {BENCH_SAMPLE_SIZE // 1024} KB")
    print(f"[INFO] 모델 경로: {MODEL_PATH}")

    # 작업 리스트 생성 (기존 benchmark_utils.build_tasks 재사용)
    tasks = build_tasks()
    print(f"[INFO] 대상 청크 수: {len(tasks)}")
    if MAX_CHUNKS is not None:
        print(f"[INFO] (MAX_CHUNKS = {MAX_CHUNKS} → 앞 {MAX_CHUNKS}개만 사용)")

    results = {}  # workers -> dict(...)

    for num_workers in NUM_WORKER_LIST:
        workers = min(num_workers, mp.cpu_count())
        print("\n" + "=" * 60)
        print(f"[실행] 모델 기반 멀티프로세싱 (workers = {workers})")
        print("=" * 60)

        total_feat = 0.0
        total_pred = 0.0
        total_comp = 0.0

        t_wall0 = time.perf_counter()

        if workers == 1:
            # 싱글 프로세스 버전
            for t in tasks:
                f_t, p_t, c_t = model_overhead_worker(t)
                total_feat += f_t
                total_pred += p_t
                total_comp += c_t
        else:
            # 멀티프로세스 버전
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for f_t, p_t, c_t in pool.imap_unordered(model_overhead_worker, tasks, chunksize=4):
                    total_feat += f_t
                    total_pred += p_t
                    total_comp += c_t

        t_wall1 = time.perf_counter()
        wall_time = t_wall1 - t_wall0

        total_time = total_feat + total_pred + total_comp

        if total_time > 0:
            feat_ratio = total_feat / total_time
            pred_ratio = total_pred / total_time
            comp_ratio = total_comp / total_time
        else:
            feat_ratio = pred_ratio = comp_ratio = 0.0

        avg_feat = total_feat / len(tasks)
        avg_pred = total_pred / len(tasks)
        avg_comp = total_comp / len(tasks)

        print(f"[workers={workers}] 청크당 평균 feat_time = {avg_feat:.6f} s")
        print(f"[workers={workers}] 청크당 평균 pred_time = {avg_pred:.6f} s")
        print(f"[workers={workers}] 청크당 평균 comp_time = {avg_comp:.6f} s")
        print(f"[workers={workers}] feat+pred+comp 합(total_time) = {total_time:.4f} s")
        print(f"[workers={workers}] 전체 벤치마크 wall time = {wall_time:.4f} s")
        print(f"[workers={workers}] 비율: feat={feat_ratio*100:.1f}%, "
              f"pred={pred_ratio*100:.1f}%, comp={comp_ratio*100:.1f}%")

        results[workers] = {
            "feat": total_feat,
            "pred": total_pred,
            "comp": total_comp,
            "feat_ratio": feat_ratio,
            "pred_ratio": pred_ratio,
            "comp_ratio": comp_ratio,
            "total_time": total_time,
            "wall_time": wall_time,
        }

    # =========================
    # 5. 그래프 생성 (workers vs 비율)
    # =========================
    workers_sorted = sorted(results.keys())
    feat_ratios = [results[w]["feat_ratio"] for w in workers_sorted]
    pred_ratios = [results[w]["pred_ratio"] for w in workers_sorted]
    comp_ratios = [results[w]["comp_ratio"] for w in workers_sorted]

    graphs_dir = GRAPH_PATH  # benchmark_config 에 이미 설정됨
    graphs_dir.mkdir(parents=True, exist_ok=True)
    out_path = graphs_dir / "model_overhead_ratio.png"

    x = np.arange(len(workers_sorted))
    bar_width = 0.6

    plt.figure(figsize=(8, 5))
    plt.bar(x, feat_ratios, width=bar_width, label="Feature extraction")
    plt.bar(x, pred_ratios, width=bar_width, bottom=feat_ratios, label="Model prediction")
    bottom_comp = np.array(feat_ratios) + np.array(pred_ratios)
    plt.bar(x, comp_ratios, width=bar_width, bottom=bottom_comp, label="Compression")

    plt.xticks(x, workers_sorted)
    plt.ylim(0, 1.0)
    plt.ylabel("Time ratio")
    plt.xlabel("Number of workers")
    plt.title("Model-based compression time breakdown (I/O excluded)")
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\n[INFO] 시간 비율 그래프 저장 완료:")
    print(f"  - {out_path}")


if __name__ == "__main__":
    main()
