#!/usr/bin/env python
"""
벤치마크 메인 스크립트

- raw/ 디렉토리의 파일들을 16MB 청크로 나눔
- 모델 기반 압축 벤치마크 (싱글/멀티)
- 단일 코덱 압축 벤치마크 (zstd, lz4, snappy)
- 결과를 CSV로 저장하고 그래프 생성
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 상대 import 사용
from benchmark_config import (
    MODEL_PATH, RAW_DIR, CHUNK_SIZE_BYTES, 
    BENCH_SAMPLE_SIZE, MAX_CHUNKS, NUM_WORKER_LIST
)
from benchmark_utils import (
    build_tasks, print_single_result, print_result_block,
    save_throughput_csv, save_weighted_score_csv
)
from benchmark_runners import (
    run_model_single, run_model_multi,
    run_codec_single, run_codec_multi
)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    raw_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if not raw_files:
        raise RuntimeError(f"raw 디렉토리에 파일이 없습니다: {RAW_DIR}")
    
    print("="*60)
    print("벤치마크 설정")
    print("="*60)
    print(f"[INFO] 모델: {MODEL_PATH}")
    print(f"[INFO] raw 디렉토리: {RAW_DIR} (파일 {len(raw_files)}개)")
    for p in raw_files:
        print(f"  - {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"\n[INFO] 청크 크기: {CHUNK_SIZE_BYTES // (1024*1024)} MB")
    print(f"[INFO] 샘플링 크기: {BENCH_SAMPLE_SIZE // 1024} KB")
    
    # 작업 리스트 생성
    tasks = build_tasks()
    print(f"[INFO] 대상 청크 수: {len(tasks)}")
    if MAX_CHUNKS is not None:
        print(f"[INFO] (MAX_CHUNKS = {MAX_CHUNKS} → 앞 {MAX_CHUNKS}개만 사용)")
    
    # 결과 저장용 딕셔너리
    all_results = {}  # {num_workers: {method: (orig, comp, time)}}
    
    # =============================
    # 싱글 프로세스 벤치마크
    # =============================
    print("\n" + "="*60)
    print("싱글 프로세스 벤치마크")
    print("="*60)
    
    # 모델 기반
    model_single = run_model_single(tasks)
    print_single_result("모델 기반", model_single)
    single_results = {'Model-based': model_single}
    
    # 단일 코덱들
    for codec in ["zstd", "lz4", "snappy"]:
        codec_single = run_codec_single(tasks, codec)
        print_single_result(f"단일 코덱: {codec}", codec_single)
        single_results[codec.upper()] = codec_single
    
    # 워커 1개는 싱글 프로세스 결과 사용
    all_results[1] = single_results.copy()
    
    # =============================
    # 멀티 프로세스 벤치마크
    # =============================
    for num_workers in NUM_WORKER_LIST:
        if num_workers == 1:
            continue
        
        print("\n" + "="*60)
        print(f"멀티 프로세스 벤치마크 (워커 {num_workers}개)")
        print("="*60)
        
        all_results[num_workers] = {}
        
        # 모델 기반
        model_multi = run_model_multi(tasks, num_workers)
        print_result_block("모델 기반 (adaptive)", model_single, model_multi)
        all_results[num_workers]['Model-based'] = model_multi
        
        # 단일 코덱들
        for codec in ["zstd", "lz4", "snappy"]:
            codec_multi = run_codec_multi(tasks, codec, num_workers)
            print_result_block(f"단일 코덱: {codec}", single_results[codec.upper()], codec_multi)
            all_results[num_workers][codec.upper()] = codec_multi
    
    # =============================
    # 결과 저장
    # =============================
    print("\n" + "="*60)
    print("결과 저장")
    print("="*60)
    
    save_throughput_csv(all_results, "throughput_results.csv")
    
    if 4 in all_results:
        save_weighted_score_csv(all_results[4], num_workers=4, filename="weighted_score_results.csv")
    
    print("\n" + "="*60)
    print("="*60)
    print("\n그래프를 생성하려면 다음 명령어를 실행하세요:")
    print("  python3 graphs/plot_results.py")


if __name__ == "__main__":
    main()