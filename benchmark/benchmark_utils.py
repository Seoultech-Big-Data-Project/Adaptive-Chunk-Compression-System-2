"""
벤치마크 유틸리티 함수들
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_config import RAW_DIR, CHUNK_SIZE_BYTES, MAX_CHUNKS, GRAPH_PATH


def build_tasks() -> list[tuple[str, int, int]]:
    """
    raw/ 디렉토리의 파일들을 청크로 분할한 작업 리스트 생성
    
    Returns:
        [(파일경로, offset, chunk_size), ...]
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


def print_single_result(name: str, result: tuple[int, int, float]):
    """싱글 프로세스 결과 출력"""
    orig, comp, elapsed = result
    mb = orig / (1024 * 1024)
    ratio = orig / comp if comp > 0 else float('inf')
    throughput = mb / elapsed if elapsed > 0 else 0
    
    print(f"\n>>> {name}")
    print(f"  - 압축률: {ratio:.4f}")
    print(f"  - 소요 시간: {elapsed:.2f} 초")
    print(f"  - 처리 속도: {throughput:.2f} MB/s")


def print_result_block(title: str, single_res: tuple[int, int, float], multi_res: tuple[int, int, float]):
    """싱글 vs 멀티 비교 결과 출력"""
    s_orig, s_comp, s_time = single_res
    m_orig, m_comp, m_time = multi_res

    m_mb = m_orig / (1024 * 1024)
    m_ratio = m_orig / m_comp if m_comp > 0 else float("inf")
    m_throughput = m_mb / m_time if m_time > 0 else 0.0
    speedup = s_time / m_time if (s_time > 0 and m_time > 0) else float("inf")

    print(f"\n>>> {title}")
    print(f"  - 압축률: {m_ratio:.4f}")
    print(f"  - 소요 시간: {m_time:.2f} 초")
    print(f"  - 처리 속도: {m_throughput:.2f} MB/s")
    print(f"  - 속도 배율 (single/multi): {speedup:.2f}x")


def save_throughput_csv(results: dict, filename: str = "throughput_results.csv"):
    """워커 수별 throughput 결과를 CSV로 저장"""
    csv_path = Path(GRAPH_PATH) / filename
    csv_path.parent.mkdir(exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['workers', 'method', 'throughput_MB_per_s', 'compression_ratio', 'time_sec'])
        
        for num_workers in sorted(results.keys()):
            for method, (orig, comp, elapsed) in results[num_workers].items():
                throughput = (orig / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                ratio = orig / comp if comp > 0 else float('inf')
                writer.writerow([num_workers, method, f"{throughput:.2f}", f"{ratio:.4f}", f"{elapsed:.2f}"])
    
    print(f"\n[INFO] Throughput 결과 저장: {csv_path}")


def save_weighted_score_csv(results: dict, num_workers: int = 4, filename: str = "weighted_score_results.csv"):
    """Weighted Score 계산 후 CSV로 저장"""
    csv_path = Path(GRAPH_PATH) / filename
    csv_path.parent.mkdir(exist_ok=True)
    
    # 각 방법의 throughput과 ratio 계산
    data = {}
    for method, (orig, comp, elapsed) in results.items():
        throughput = (orig / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        ratio = orig / comp if comp > 0 else float('inf')
        data[method] = {'throughput': throughput, 'ratio': ratio}
    
    # 최대값 찾기
    max_speed = max(d['throughput'] for d in data.values())
    max_ratio = max(d['ratio'] for d in data.values())
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['speed_weight', 'method', 'throughput', 'ratio', 'norm_speed', 'norm_ratio', 'weighted_score'])
        
        # speed_weight 0~10까지
        for s in range(11):
            alpha = s / 10
            beta = (10 - s) / 10
            
            for method, values in data.items():
                norm_speed = values['throughput'] / max_speed if max_speed > 0 else 0
                norm_ratio = values['ratio'] / max_ratio if max_ratio > 0 else 0
                weighted_score = alpha * norm_speed + beta * norm_ratio
                
                writer.writerow([
                    s, 
                    method, 
                    f"{values['throughput']:.2f}", 
                    f"{values['ratio']:.4f}",
                    f"{norm_speed:.4f}",
                    f"{norm_ratio:.4f}",
                    f"{weighted_score:.4f}"
                ])
    
    print(f"[INFO] Weighted Score 결과 저장: {csv_path}")
