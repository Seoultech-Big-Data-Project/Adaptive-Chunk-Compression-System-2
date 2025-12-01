#!/usr/bin/env python
"""
벤치마크 결과 그래프 생성 스크립트
1. 워커 수별 throughput 비교 그래프
2. Speed weight별 weighted score 그래프
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

plt.rcParams['axes.unicode_minus'] = False

def plot_throughput_graph(csv_file: str = "throughput_results.csv"):
    """
    워커 수별 throughput 비교 그래프 생성
    """
    csv_path = Path(__file__).parent / csv_file
    
    if not csv_path.exists():
        print(f"[ERROR] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # CSV 읽기
    data = {}  # {method: {workers: throughput}}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            workers = int(row['workers'])
            throughput = float(row['throughput_MB_per_s'])
            
            if method not in data:
                data[method] = {}
            data[method][workers] = throughput
    
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    
    colors = {
        'Model-based': '#2E86AB',
        'ZSTD': '#A23B72',
        'LZ4': '#F18F01',
        'SNAPPY': '#C73E1D'
    }
    
    markers = {
        'Model-based': 'o',
        'ZSTD': 's',
        'LZ4': '^',
        'SNAPPY': 'D'
    }
    
    # Model-based를 맨 앞에 오도록 정렬
    method_order = ['Model-based', 'LZ4', 'SNAPPY', 'ZSTD']
    sorted_methods = sorted(data.items(), key=lambda x: method_order.index(x[0]) if x[0] in method_order else 999)
    
    for method, worker_data in sorted_methods:
        workers = sorted(worker_data.keys())
        throughputs = [worker_data[w] for w in workers]
        
        plt.plot(workers, throughputs, 
                marker=markers.get(method, 'o'), 
                linewidth=2, 
                markersize=8,
                color=colors.get(method, 'black'),
                label=method)
    
    plt.xlabel('Number of Workers', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (MB/s)', fontsize=12, fontweight='bold')
    plt.title('Parallel Compression Throughput by Number of Workers', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(workers)
    
    # 저장
    output_path = Path(__file__).parent / "throughput_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Throughput 그래프 저장: {output_path}")
    plt.close()


def plot_weighted_score_graph(csv_file: str = "weighted_score_results.csv"):
    """
    Speed weight별 weighted score 그래프 생성
    """
    csv_path = Path(__file__).parent / csv_file
    
    if not csv_path.exists():
        print(f"[ERROR] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # CSV 읽기
    data = {}  # {method: {speed_weight: weighted_score}}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            speed_weight = int(row['speed_weight'])
            weighted_score = float(row['weighted_score'])
            
            if method not in data:
                data[method] = {}
            data[method][speed_weight] = weighted_score
    
    # 그래프 생성
    plt.figure(figsize=(12, 6))
    
    colors = {
        'Model-based': '#2E86AB',
        'ZSTD': '#A23B72',
        'LZ4': '#F18F01',
        'SNAPPY': '#C73E1D'
    }
    
    markers = {
        'Model-based': 'o',
        'ZSTD': 's',
        'LZ4': '^',
        'SNAPPY': 'D'
    }
    
    method_order = ['Model-based', 'LZ4', 'SNAPPY', 'ZSTD']
    sorted_methods = sorted(data.items(), key=lambda x: method_order.index(x[0]) if x[0] in method_order else 999)
    
    for method, weight_data in sorted_methods:
        weights = sorted(weight_data.keys())
        scores = [weight_data[w] for w in weights]
        
        plt.plot(weights, scores,
                marker=markers.get(method, 'o'),
                linewidth=2.5,
                markersize=8,
                color=colors.get(method, 'black'),
                label=method,
                alpha=0.9)
    
    plt.xlabel('Speed Weight', fontsize=12, fontweight='bold')
    plt.ylabel('Weighted Score', fontsize=12, fontweight='bold')
    plt.title('Speed Weight vs Weighted Score Comparison', 
              fontsize=13, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(range(0, 11))
    plt.xlim(-0.5, 10.5)
    
    # 저장
    output_path = Path(__file__).parent / "weighted_score_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Weighted Score 그래프 저장: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("벤치마크 결과 그래프 생성")
    print("="*60)
    
    # 1. Throughput 그래프
    print("\n[1] Throughput 비교 그래프 생성 중...")
    plot_throughput_graph()
    
    # 2. Weighted Score 그래프
    print("\n[2] Weighted Score 그래프 생성 중...")
    plot_weighted_score_graph()
    
    print("\n" + "="*60)
    print("그래프 생성 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
