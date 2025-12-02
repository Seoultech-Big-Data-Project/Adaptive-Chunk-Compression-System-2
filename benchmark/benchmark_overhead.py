# benchmark/benchmark_overhead.py

#!/usr/bin/env python
"""
오버헤드 분석 통합 스크립트

- raw/ 디렉토리의 파일들을 CHUNK_SIZE_BYTES 단위로 분할
- 다음 방법들에 대해 오버헤드 측정:
    1) 모델 기반 (single)
    2) 모델 기반 (multi, NUM_WORKER_LIST)
    3) 단일 코덱 (multi, NUM_WORKER_LIST × CODECS)

- 각 청크에 대해:
    - 파일 읽기 시간 (t_read)
    - 피처 추출 시간 (t_feature)   ← 모델만 의미 있음
    - 모델 예측 시간 (t_predict)    ← 모델만 의미 있음
    - 실제 압축 시간 (t_compress)
    - 전체 함수 실행 시간 (t_total)

- 결과를 CSV로 저장:
    benchmark/graphs/model_overhead_results.csv

- 동시에 방법(method)별 평균 / 비율 요약 출력
"""

import sys
import csv
import multiprocessing as mp
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_config import (
    RAW_DIR,
    MODEL_PATH,
    CHUNK_SIZE_BYTES,
    MAX_CHUNKS,
    GRAPH_PATH,
    NUM_WORKER_LIST,
    CODECS,
)
from benchmark_utils import build_tasks
from benchmark_workers import (
    model_compress_worker_timed,
    codec_compress_worker_timed,
)

OUTPUT_CSV = GRAPH_PATH / "overhead_results_all.csv"


# ============================
# 공통 집계 유틸
# ============================

def aggregate_rows(rows):
    """
    rows: [
        (
            file_path_str, offset, chunk_size, orig_size,
            t_read, t_feature, t_predict, t_compress, t_total, codec
        ), ...
    ]
    """
    total_bytes = 0
    n_chunks = 0
    sum_t_read = 0.0
    sum_t_feature = 0.0
    sum_t_predict = 0.0
    sum_t_compress = 0.0
    sum_t_total = 0.0

    for (
        file_path_str,
        offset,
        chunk_size,
        orig_size,
        t_read,
        t_feature,
        t_predict,
        t_compress,
        t_total,
        codec,
    ) in rows:
        if orig_size <= 0:
            continue
        total_bytes += orig_size
        n_chunks += 1
        sum_t_read += t_read
        sum_t_feature += t_feature
        sum_t_predict += t_predict
        sum_t_compress += t_compress
        sum_t_total += t_total

    if n_chunks == 0:
        return {
            "n_chunks": 0,
            "total_bytes": 0,
            "avg_read": 0.0,
            "avg_feature": 0.0,
            "avg_predict": 0.0,
            "avg_compress": 0.0,
            "avg_total": 0.0,
            "throughput_MBps": 0.0,
        }

    avg_read = sum_t_read / n_chunks
    avg_feature = sum_t_feature / n_chunks
    avg_predict = sum_t_predict / n_chunks
    avg_compress = sum_t_compress / n_chunks
    avg_total = sum_t_total / n_chunks

    mb_total = total_bytes / (1024 * 1024)
    throughput = mb_total / sum_t_total if sum_t_total > 0 else 0.0

    return {
        "n_chunks": n_chunks,
        "total_bytes": total_bytes,
        "avg_read": avg_read,
        "avg_feature": avg_feature,
        "avg_predict": avg_predict,
        "avg_compress": avg_compress,
        "avg_total": avg_total,
        "throughput_MBps": throughput,
    }


# ============================
# 실행 함수들
# ============================

def run_model_single_timed(tasks):
    rows = []
    for task in tasks:
        rows.append(model_compress_worker_timed(task))
    stats = aggregate_rows(rows)
    return rows, stats


def run_model_multi_timed(tasks, num_workers: int):
    rows = []
    workers = min(mp.cpu_count(), num_workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(model_compress_worker_timed, tasks, chunksize=4):
            rows.append(res)
    stats = aggregate_rows(rows)
    return rows, stats


def run_codec_multi_timed(tasks, codec: str, num_workers: int):
    rows = []
    workers = min(mp.cpu_count(), num_workers)
    ctx = mp.get_context("spawn")

    def args_iter():
        for (file_path_str, offset, chunk_size) in tasks:
            yield (codec, file_path_str, offset, chunk_size)

    with ctx.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(codec_compress_worker_timed, args_iter(), chunksize=4):
            rows.append(res)
    stats = aggregate_rows(rows)
    return rows, stats


# ============================
# 메인
# ============================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    raw_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if not raw_files:
        raise RuntimeError(f"raw 디렉토리에 파일이 없습니다: {RAW_DIR}")

    print("=" * 60)
    print("오버헤드 분석 (모델 기반 + 단일 코덱 멀티)")
    print("=" * 60)
    print(f"[INFO] 모델: {MODEL_PATH}")
    print(f"[INFO] raw 디렉토리: {RAW_DIR} (파일 {len(raw_files)}개)")
    for p in raw_files:
        print(f"  - {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"\n[INFO] 청크 크기: {CHUNK_SIZE_BYTES // (1024*1024)} MB")
    if MAX_CHUNKS is not None:
        print(f"[INFO] 최대 청크 수 제한: {MAX_CHUNKS}")

    # 작업 리스트
    tasks = build_tasks()
    num_tasks = len(tasks)
    print(f"[INFO] 대상 청크 수: {num_tasks}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT_CSV.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    # 공통 CSV 헤더
    writer.writerow([
        "method",
        "workers",
        "file_name",
        "file_path",
        "offset",
        "chunk_size",
        "orig_size",
        "codec_used",
        "t_read",
        "t_feature",
        "t_predict",
        "t_compress",
        "t_total",
    ])

    # 요약 저장용
    summary = []

    def write_rows(method_name: str, workers: int, rows):
        for (
            file_path_str,
            offset,
            chunk_size,
            orig_size,
            t_read,
            t_feature,
            t_predict,
            t_compress,
            t_total,
            codec_used,
        ) in rows:
            file_name = Path(file_path_str).name
            writer.writerow([
                method_name,
                workers,
                file_name,
                file_path_str,
                offset,
                chunk_size,
                orig_size,
                codec_used,
                f"{t_read:.6f}",
                f"{t_feature:.6f}",
                f"{t_predict:.6f}",
                f"{t_compress:.6f}",
                f"{t_total:.6f}",
            ])

    print("\n[1] 모델 기반(single) 오버헤드 측정...")
    model_single_rows, model_single_stats = run_model_single_timed(tasks)
    write_rows("Model_single", 1, model_single_rows)
    summary.append(("Model_single", 1, model_single_stats))

    # 멀티프로세스: 모델 + 단일코덱
    for n_workers in NUM_WORKER_LIST:
        if n_workers < 2:
            continue

        print(f"\n[2] 모델 기반(multi, workers={n_workers}) 오버헤드 측정...")
        m_rows, m_stats = run_model_multi_timed(tasks, n_workers)
        write_rows(f"Model_multi", n_workers, m_rows)
        summary.append((f"Model_multi", n_workers, m_stats))

        for codec in CODECS:
            print(f"\n[3] 단일 코덱 '{codec}' (multi, workers={n_workers}) 오버헤드 측정...")
            c_rows, c_stats = run_codec_multi_timed(tasks, codec, n_workers)
            method_name = f"{codec.upper()}_multi"
            write_rows(method_name, n_workers, c_rows)
            summary.append((method_name, n_workers, c_stats))

    csv_file.close()
    print(f"\n[INFO] 전체 오버헤드 결과 CSV 저장 완료: {OUTPUT_CSV}")

    # ============================
    # 요약 출력
    # ============================

    print("\n" + "=" * 60)
    print("방법(method)별 평균 시간 / 비율 요약 (per chunk)")
    print("=" * 60)

    def pct(part, total):
        return (part / total * 100.0) if total > 0 else 0.0

    # 보기 좋게 정렬 (workers, method)
    summary_sorted = sorted(summary, key=lambda x: (x[1], x[0]))

    for method_name, workers, stats in summary_sorted:
        n_chunks = stats["n_chunks"]
        if n_chunks == 0:
            continue

        avg_read = stats["avg_read"]
        avg_feature = stats["avg_feature"]
        avg_predict = stats["avg_predict"]
        avg_compress = stats["avg_compress"]
        avg_total = stats["avg_total"]
        throughput = stats["throughput_MBps"]
        mb_total = stats["total_bytes"] / (1024 * 1024)

        print(f"\n--- {method_name} (workers={workers}) ---")
        print(f"  · 청크 수              : {n_chunks}")
        print(f"  · 총 데이터 크기       : {mb_total:.2f} MB")
        print(f"  · 전체 처리 속도       : {throughput:.2f} MB/s")

        print(f"  [평균 시간 / 청크 기준]")
        print(f"    · 파일 읽기       : {avg_read:.6f} s  ({pct(avg_read, avg_total):5.1f} %)")
        print(f"    · 피처 추출       : {avg_feature:.6f} s  ({pct(avg_feature, avg_total):5.1f} %)")
        print(f"    · 모델 예측       : {avg_predict:.6f} s  ({pct(avg_predict, avg_total):5.1f} %)")
        print(f"    · 실제 압축       : {avg_compress:.6f} s  ({pct(avg_compress, avg_total):5.1f} %)")
        print(f"    · 전체(함수 기준) : {avg_total:.6f} s  (100.0 %)")
        print()

    print("=" * 60)
    print("해석 팁")
    print("=" * 60)
    print("  - 같은 workers 수에서 'Model_multi' vs 'ZSTD_multi/LZ4_multi/SNAPPY_multi'를 비교하면")
    print("    파일 읽기 비율이 정말 비슷한지, 모델 쪽에서 추가로 드는 피처/예측 오버헤드가")
    print("    어느 정도인지 정확히 볼 수 있음.")
    print("  - 파일 읽기 비율이 서로 크게 다르다면, 디스크 I/O 패턴(랜덤/순차),")
    print("    워커 스케줄링, OS 캐시 등에 의한 차이일 수 있고, 그때는 작업 분배/순서를")
    print("    조정하는 쪽 최적화를 고려해볼 수 있음.")


if __name__ == "__main__":
    main()
