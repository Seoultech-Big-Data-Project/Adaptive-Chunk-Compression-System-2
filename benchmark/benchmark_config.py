"""
벤치마크 설정 파일
"""
from pathlib import Path
import sys

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 디렉토리 및 모델 경로
RAW_DIR = PROJECT_ROOT / "raw"
MODEL_PATH = PROJECT_ROOT / "models" / "model_sample_128KB.json"
GRAPH_PATH = PROJECT_ROOT / "benchmark" / "graphs"

# 청크 및 샘플링 설정
CHUNK_SIZE_BYTES = 16 * 1024 * 1024 
BENCH_SAMPLE_SIZE = 128 * 1024

# 최대 청크 수 제한 (None이면 전체)
MAX_CHUNKS = None

# 워커 수 리스트
NUM_WORKER_LIST = [1, 2, 4, 8]

# 코덱 설정
CODECS = ["zstd", "lz4", "snappy"]
CODEC_TO_LABEL = {c: i for i, c in enumerate(CODECS)}
LABEL_TO_CODEC = {i: c for i, c in enumerate(CODECS)}