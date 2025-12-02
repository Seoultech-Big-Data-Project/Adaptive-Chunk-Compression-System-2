#!/usr/bin/env python
"""
codec_ratio.py

raw/ 디렉토리의 생 데이터 파일들을 여러 청크 사이즈로 나누고,
각 청크에 대해 XGBoost 모델로 "최적 코덱"을 예측한 뒤

- 청크 사이즈별
- 코덱별 (zstd / lz4 / snappy)

예측 비율을 출력하고 CSV로 저장하는 스크립트.

샘플링은 features.sample_bytes 를 사용하며, max_len=32KB 기준.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import xgboost as xgb
import csv

# =====================================
# 경로 / 설정
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 내부 모듈 import (이미 benchmark에서 쓰던 것들 그대로 재사용)
from benchmark.benchmark_config import RAW_DIR, MODEL_PATH, LABEL_TO_CODEC, CODECS  # type: ignore
from features.features import sample_bytes, extract_all_features, FEATURE_KEYS, FEATURE_COLUMNS  # type: ignore

# 샘플링 크기: 32KB
SAMPLE_BYTES = 32 * 1024

# 분석할 청크 사이즈 리스트 (원하면 여기 값들 조절해서 쓰면 됨)
CHUNK_SIZES_BYTES = [
    1 * 1024 * 1024,   # 1MB
    2 * 1024 * 1024,   # 2MB
    4 * 1024 * 1024,   # 4MB
    8 * 1024 * 1024,   # 8MB
    16 * 1024 * 1024,  # 16MB
    32 * 1024 * 1024,  # 32MB
]

# 결과 저장 경로
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "codec_ratio_by_chunk_raw.csv"


def build_tasks_for_chunk_size(chunk_size_bytes: int) -> list[tuple[str, int, int]]:
    """
    주어진 chunk_size_bytes 에 대해
    raw/ 디렉토리의 파일을 (file_path, offset, chunk_size) 리스트로 쪼갠다.
    """
    tasks: list[tuple[str, int, int]] = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR 이 존재하지 않습니다: {RAW_DIR}")

    raw_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if not raw_files:
        raise RuntimeError(f"raw 디렉토리에 파일이 없습니다: {RAW_DIR}")

    for file_path in raw_files:
        size = file_path.stat().st_size
        offset = 0
        while offset < size:
            tasks.append((str(file_path), offset, chunk_size_bytes))
            offset += chunk_size_bytes

    return tasks


def load_model() -> xgb.Booster:
    """
    XGBoost 모델 로드
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


def predict_codec_for_chunk(
    booster: xgb.Booster,
    data: bytes,
    chunk_size_bytes: int,
) -> str:
    """
    한 청크에 대해 샘플링 → 피처 추출 → 모델 예측 → 코덱 이름 반환
    """
    if len(data) == 0:
        return "empty"

    # 1) 샘플링 (32KB 기준)
    sampled = sample_bytes(data, max_len=SAMPLE_BYTES)

    # 2) 피처 추출
    feats = extract_all_features(sampled)

    # 3) 피처 벡터 + chunk_size_bytes
    feat_vec = np.array(
        [[feats[k] for k in FEATURE_KEYS] + [float(chunk_size_bytes)]],
        dtype=np.float32,
    )
    dmat = xgb.DMatrix(feat_vec, feature_names=FEATURE_COLUMNS)

    # 4) 예측
    prob = booster.predict(dmat)[0]
    if prob.ndim == 0:
        # 혹시 binary 형태로 나올 경우 대비 (여기는 보통 multi-class)
        pred_label = int(prob > 0.5)
    else:
        pred_label = int(prob.argmax())

    pred_codec = LABEL_TO_CODEC.get(pred_label, "unknown")
    return pred_codec


def main():
    print("=" * 60)
    print("청크 사이즈별 모델 예측 코덱 비율 분석 (raw 기반)")
    print("=" * 60)

    # 모델 로드
    booster = load_model()
    print(f"[INFO] 모델 로드 완료: {MODEL_PATH}")

    # 결과 CSV 준비
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "chunk_size_bytes",
            "chunk_size_MB",
            "codec",
            "count",
            "ratio",
            "total_chunks",
        ])

        # 청크 사이즈별로 반복
        for chunk_size in CHUNK_SIZES_BYTES:
            chunk_size_mb = chunk_size / (1024 * 1024)
            print("\n" + "-" * 60)
            print(f"청크 사이즈 = {chunk_size_mb:.0f} MB")
            print("-" * 60)

            # 작업 리스트 생성
            tasks = build_tasks_for_chunk_size(chunk_size)
            num_tasks = len(tasks)
            print(f"[INFO] 대상 청크 수: {num_tasks}")

            codec_counter: Counter[str] = Counter()
            total_chunks = 0

            # 각 청크에 대해 예측
            for idx, (file_path_str, offset, chunk_sz) in enumerate(tasks, start=1):
                file_path = Path(file_path_str)
                with file_path.open("rb") as f:
                    f.seek(offset)
                    data = f.read(chunk_sz)

                if not data:
                    continue

                pred_codec = predict_codec_for_chunk(booster, data, chunk_sz)
                codec_counter[pred_codec] += 1
                total_chunks += 1

                if idx % 100 == 0 or idx == num_tasks:
                    print(f"  - 진행 상황: {idx}/{num_tasks} 청크 처리 완료", end="\r")

            print()  # 줄바꿈

            if total_chunks == 0:
                print("[WARN] 유효한 청크가 없습니다. (데이터가 비어있는 경우)")
                continue

            print(f"[INFO] 총 유효 청크 수: {total_chunks}")

            # 코덱 리스트: 모델에서 쓰는 CODECS 기준 + 기타 (unknown/empty) 있으면 같이 출력
            all_codecs = set(CODECS) | set(codec_counter.keys())

            print("[코덱별 예측 비율]")
            for codec in sorted(all_codecs):
                cnt = codec_counter.get(codec, 0)
                ratio = cnt / total_chunks
                print(f"  - {codec:7s}: {ratio:7.3f}  ({cnt}개)")
                writer.writerow([
                    chunk_size,
                    f"{chunk_size_mb:.0f}",
                    codec,
                    cnt,
                    f"{ratio:.6f}",
                    total_chunks,
                ])

    print("\n" + "=" * 60)
    print(f"[INFO] 결과 CSV 저장 완료: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
