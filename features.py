#!/usr/bin/env python
"""
features.py

청크 바이트 데이터에서 피처를 추출하는 공통 모듈.
- sample_bytes
- compute_features_arr_numpy
- extract_all_features
- FEATURE_KEYS
- FEATURE_COLUMNS
"""

from __future__ import annotations

import numpy as np

# 학습/추론에서 사용할 피처 목록
FEATURE_KEYS = [
    "entropy",
    "frac_zero",
    "frac_ff",
    "frac_ascii_printable",
    "frac_control",
    "frac_space",
    "frac_newline",
    "run_mean",
    "run_std",
    "runs_per_byte",
]

# 마지막에 chunk_size_bytes를 붙여서 모델 입력으로 사용
FEATURE_COLUMNS = FEATURE_KEYS + ["chunk_size_bytes"]


def sample_bytes(data: bytes, max_len: int) -> bytes:
    """
    큰 청크에서 max_len 정도만 균일 샘플링.
    - 압축 자체는 full chunk 기준으로 하고
    - 피처 계산만 이 샘플로 수행할 때 사용.
    """
    n = len(data)
    if n <= max_len:
        return data
    stride = n // max_len
    return data[::stride]


def compute_features_arr_numpy(arr: np.ndarray) -> np.ndarray:
    """
    uint8 1D 배열(arr)에서 피처 벡터(길이 10)를 계산.
      0: entropy
      1: frac_zero
      2: frac_ff
      3: frac_ascii_printable
      4: frac_control
      5: frac_space
      6: frac_newline
      7: run_mean
      8: run_std
      9: runs_per_byte
    """
    length = arr.size
    out = np.zeros(10, dtype=np.float64)
    if length == 0:
        return out

    length_f = float(length)

    # 1) 0~255 counts
    counts = np.bincount(arr, minlength=256)

    zero = counts[0]
    ff = counts[255]
    ascii_printable = counts[32:127].sum()         # 0x20 ~ 0x7E
    control = counts[:32].sum() - counts[10]       # 0x00~0x1F, 개행 제외
    space = counts[32]
    newline = counts[10]

    frac_zero = zero / length_f
    frac_ff = ff / length_f
    frac_ascii_printable = ascii_printable / length_f
    frac_control = control / length_f
    frac_space = space / length_f
    frac_newline = newline / length_f

    # 2) 엔트로피
    p = counts[counts > 0] / length_f
    ent = -np.sum(p * np.log2(p))

    # 3) run-length (벡터화)
    if length == 1:
        runs = np.array([1], dtype=np.int64)
    else:
        changes = np.nonzero(np.diff(arr) != 0)[0] + 1
        idx = np.concatenate(([0], changes, [length]))
        runs = np.diff(idx)

    num_runs = runs.size
    run_mean = runs.mean()
    run_std = runs.std()
    runs_per_byte = num_runs / length_f

    out[:] = [
        ent,
        frac_zero,
        frac_ff,
        frac_ascii_printable,
        frac_control,
        frac_space,
        frac_newline,
        run_mean,
        run_std,
        runs_per_byte,
    ]
    return out


def extract_all_features(data: bytes) -> dict[str, float]:
    """
    bytes → uint8 array → 피처 dict(FEATURE_KEYS) 로 변환.
    """
    if len(data) == 0:
        return {k: 0.0 for k in FEATURE_KEYS}
    arr = np.frombuffer(data, dtype=np.uint8)
    vals = compute_features_arr_numpy(arr)
    return {key: float(vals[i]) for i, key in enumerate(FEATURE_KEYS)}


if __name__ == "__main__":
    # 간단한 테스트용
    test = b"AAAAABBBBBCCCCCDDDDDEEEEE\n\n\n"
    feats = extract_all_features(test)
    print("features:", feats)
