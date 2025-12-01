"""
벤치마크 워커 함수들
"""
from __future__ import annotations

from pathlib import Path
import zstandard as zstd
import lz4.frame
import snappy
import xgboost as xgb
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.features import sample_bytes, extract_all_features, FEATURE_KEYS, FEATURE_COLUMNS
from benchmark_config import MODEL_PATH, BENCH_SAMPLE_SIZE, LABEL_TO_CODEC

# 프로세스별 lazy init
_MODEL: xgb.Booster | None = None
_ZSTD = None


def _compress_data(data: bytes, codec: str) -> bytes:
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


def model_compress_worker(task: tuple[str, int, int]) -> tuple[int, int]:
    """
    단일 청크에 대해 모델 기반 압축 수행
    
    Args:
        task: (파일경로, offset, chunk_size)
    
    Returns:
        (원본 크기, 압축 후 크기)
    """
    global _MODEL, _ZSTD

    file_path_str, offset, chunk_size = task
    file_path = Path(file_path_str)

    # 파일 읽기
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

    # 샘플링 + 피처 추출
    sampled = sample_bytes(data, max_len=BENCH_SAMPLE_SIZE)
    feats = extract_all_features(sampled)
    feat_vec = np.array(
        [[feats[k] for k in FEATURE_KEYS] + [float(orig_size)]],
        dtype=np.float32,
    )
    drow = xgb.DMatrix(feat_vec, feature_names=FEATURE_COLUMNS)

    # 모델 예측
    prob = _MODEL.predict(drow)[0]
    pred_label = int(prob.argmax())
    pred_codec = LABEL_TO_CODEC.get(pred_label, "lz4")

    # 압축
    comp = _compress_data(data, pred_codec)
    
    return orig_size, len(comp)


def codec_compress_worker(args: tuple[str, str, int, int]) -> tuple[int, int]:
    """
    단일 청크에 대해 특정 코덱으로 압축 수행
    
    Args:
        args: (codec, 파일경로, offset, chunk_size)
    
    Returns:
        (원본 크기, 압축 후 크기)
    """
    global _ZSTD

    codec, file_path_str, offset, chunk_size = args
    file_path = Path(file_path_str)

    # 파일 읽기
    with file_path.open("rb") as f:
        f.seek(offset)
        data = f.read(chunk_size)

    orig_size = len(data)
    if orig_size == 0:
        return 0, 0

    # 압축
    comp = _compress_data(data, codec)
    
    return orig_size, len(comp)
