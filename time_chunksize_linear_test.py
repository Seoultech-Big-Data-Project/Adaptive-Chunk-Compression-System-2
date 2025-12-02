import os
import time
from pathlib import Path
from typing import List, Dict, Any, Iterable

import matplotlib.pyplot as plt
import zstandard as zstd
import lz4.frame
import snappy


# =========================
# 설정
# =========================
RAW_DIR = Path("raw")  # 프로젝트 루트 기준 ./raw
CHUNK_SIZES_MB = [1, 2, 4, 8, 16]
REPEAT = 1  # 한 청크에 대해 몇 번 반복 측정할지 (1이면 한 번만)


# =========================
# raw 전체에서 청크 단위로 읽기
# =========================
def iter_chunks_over_raw(raw_dir: Path, chunk_size: int) -> Iterable[bytes]:
    """
    raw 디렉토리 안의 파일들을 이름 순으로 전부 읽으면서,
    chunk_size 바이트 단위로 잘라서 순차적으로 yield.

    마지막에 남는 조각이 chunk_size보다 작더라도 그냥 yield 함.
    """
    files = sorted(
        [p for p in raw_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    if not files:
        raise FileNotFoundError(f"{raw_dir} 안에 파일이 없습니다.")

    buffer = bytearray()
    for file in files:
        with file.open("rb") as f:
            while True:
                need = chunk_size - len(buffer)
                if need <= 0:
                    # 버퍼가 이미 가득 찼으면 바로 yield
                    yield bytes(buffer)
                    buffer.clear()
                    need = chunk_size

                chunk = f.read(need)
                if not chunk:  # 이 파일은 다 읽음
                    break

                buffer.extend(chunk)

                if len(buffer) == chunk_size:
                    # 정확히 chunk_size만큼 찼을 때
                    yield bytes(buffer)
                    buffer.clear()

    # 마지막에 남은 데이터(청크 사이즈보다 작음)
    if buffer:
        yield bytes(buffer)


# =========================
# 시간 측정 유틸
# =========================
def time_compress(fn, data: bytes, repeat: int = 1) -> Dict[str, Any]:
    """
    fn: data -> compressed_bytes
    repeat번 압축해서 평균 시간과 마지막 compressed size 반환
    """
    times = []
    compressed = b""
    for _ in range(repeat):
        start = time.perf_counter()
        compressed = fn(data)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return {
        "time_sec": avg_time,
        "compressed_size": len(compressed),
    }


# =========================
# 메인 벤치마크 로직
# =========================
def run_benchmark() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # 코덱별 압축 함수 정의
    zstd_compressor = zstd.ZstdCompressor(level=3)

    def compress_zstd(data: bytes) -> bytes:
        return zstd_compressor.compress(data)

    def compress_lz4(data: bytes) -> bytes:
        return lz4.frame.compress(data)

    def compress_snappy(data: bytes) -> bytes:
        return snappy.compress(data)

    codecs = {
        "zstd": compress_zstd,
        "lz4": compress_lz4,
        "snappy": compress_snappy,
    }

    print(f"[INFO] raw dir: {RAW_DIR.resolve()}")

    total_raw_size = sum(p.stat().st_size for p in RAW_DIR.glob("*") if p.is_file())
    print(f"[INFO] total raw size: {total_raw_size / (1024*1024):.2f} MB")

    for mb in CHUNK_SIZES_MB:
        chunk_size = mb * 1024 * 1024
        print(f"\n===== CHUNK SIZE: {mb} MB ({chunk_size} bytes) =====")

        # 전체 raw를 이 청크 크기로 끝까지 순회
        chunk_index = 0
        for chunk in iter_chunks_over_raw(RAW_DIR, chunk_size):
            chunk_index += 1
            orig_size = len(chunk)

            # 혹시 마지막 청크가 너무 작으면(예: 1KB) 버리고 싶다면 조건 걸어도 됨
            # if orig_size < chunk_size // 10:
            #     continue

            for codec_name, fn in codecs.items():
                try:
                    res = time_compress(fn, chunk, repeat=REPEAT)
                except Exception as e:
                    print(f"  [ERROR] codec={codec_name}, chunk_idx={chunk_index}: {e}")
                    continue

                time_sec = res["time_sec"]
                compressed_size = res["compressed_size"]
                ratio = compressed_size / orig_size  # 압축비 (작을수록 좋음)

                results.append(
                    {
                        "codec": codec_name,
                        "chunk_mb": mb,         # 의도한 청크 크기
                        "chunk_index": chunk_index,
                        "time_sec": time_sec,
                        "orig_size": orig_size,
                        "compressed_size": compressed_size,
                        "ratio": ratio,
                    }
                )

            # 진행 상황 가끔 찍어주기
            if chunk_index % 100 == 0:
                print(f"  [INFO] chunk {chunk_index} done for chunk_size={mb}MB")

        print(f"  [SUMMARY] chunk_size={mb}MB, total_chunks={chunk_index}")

    return results


# =========================
# Plot 함수
# =========================
def plot_results(results: List[Dict[str, Any]]):
    codecs = sorted(set(r["codec"] for r in results))

    # ---------- 1) 시간 vs 청크 크기 ----------
    plt.figure()
    for codec in codecs:
        # scatter: 모든 청크별 점
        xs = [r["chunk_mb"] for r in results if r["codec"] == codec]
        ys = [r["time_sec"] for r in results if r["codec"] == codec]
        plt.scatter(xs, ys, alpha=0.2, s=10)  # 점 많으니 작게 & 투명하게

        # 평균: 청크 사이즈별 평균 시간
        mean_xs = []
        mean_ys = []
        for mb in CHUNK_SIZES_MB:
            vals = [
                r["time_sec"]
                for r in results
                if r["codec"] == codec and r["chunk_mb"] == mb
            ]
            if not vals:
                continue
            mean_xs.append(mb)
            mean_ys.append(sum(vals) / len(vals))
        plt.plot(mean_xs, mean_ys, marker="o", label=f"{codec} mean")

    plt.xlabel("Chunk size (MB)")
    plt.ylabel("Compression time (sec)")
    plt.title("Per-chunk Compression Time vs Chunk Size\n(scatter = all chunks, line = mean)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("full_raw_time_vs_chunk.png")

    # ---------- 2) 압축비 vs 청크 크기 ----------
    plt.figure()
    for codec in codecs:
        xs = [r["chunk_mb"] for r in results if r["codec"] == codec]
        ys = [r["ratio"] for r in results if r["codec"] == codec]
        plt.scatter(xs, ys, alpha=0.2, s=10)

        mean_xs = []
        mean_ys = []
        for mb in CHUNK_SIZES_MB:
            vals = [
                r["ratio"]
                for r in results
                if r["codec"] == codec and r["chunk_mb"] == mb
            ]
            if not vals:
                continue
            mean_xs.append(mb)
            mean_ys.append(sum(vals) / len(vals))
        plt.plot(mean_xs, mean_ys, marker="o", label=f"{codec} mean")

    plt.xlabel("Chunk size (MB)")
    plt.ylabel("Compression ratio (compressed / original)")
    plt.title("Per-chunk Compression Ratio vs Chunk Size\n(scatter = all chunks, line = mean)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("full_raw_ratio_vs_chunk.png")

    print("\n[INFO] 그래프 저장 완료:")
    print("  - full_raw_time_vs_chunk.png")
    print("  - full_raw_ratio_vs_chunk.png")


if __name__ == "__main__":
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} 디렉토리 {RAW_DIR.resolve()} 가 없습니다.")

    all_results = run_benchmark()
    plot_results(all_results)
    print("[DONE] benchmark finished.")
