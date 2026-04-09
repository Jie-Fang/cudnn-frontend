"""
Benchmark MoE Grouped Matmul with LLM-inspired shapes.

Usage:
    python benchmarks/bench_moe.py
"""

import argparse
import statistics
from typing import Callable, List, Tuple

import torch


def benchmark_fn(
    fn: Callable, *args, warmup: int = 10, repeat: int = 100, **kw
) -> Tuple[float, float]:
    for _ in range(warmup):
        fn(*args, **kw)
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args, **kw)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times), min(times)


MOE_MODELS = {
    "mixtral-8x7b": {
        "hidden": 4096,
        "intermediate": 14336,
        "num_experts": 8,
        "top_k": 2,
    },
    "deepseek-v3": {
        "hidden": 7168,
        "intermediate": 18432,
        "num_experts": 64,
        "top_k": 8,
    },
}

TOKEN_COUNTS = [1024, 4096, 16384]


def bench_moe(warmup: int, repeat: int):
    print()
    print("=" * 100)
    print("  MoE Grouped Matmul  (FFN projections in MoE models)")
    print("=" * 100)

    try:
        from cudnn.experimental.ops import moe_grouped_matmul
    except Exception as e:
        print(f"  [SKIP] {e}")
        return

    W = [18, 10, 10, 8, 14, 14, 10]
    print(
        "  ".join(
            c.ljust(w)
            for c, w in zip(
                [
                    "model/proj",
                    "tokens",
                    "experts",
                    "topk",
                    "cuDNN (us)",
                    "Naive (us)",
                    "speedup",
                ],
                W,
            )
        )
    )
    print("-" * 95)

    for name, cfg in MOE_MODELS.items():
        hidden = cfg["hidden"]
        intermediate = cfg["intermediate"]
        num_experts = cfg["num_experts"]
        top_k = cfg["top_k"]

        for proj, K_dim, N_dim in [
            ("up", hidden, intermediate),
            ("down", intermediate, hidden),
        ]:
            for n_tokens in TOKEN_COUNTS:
                total = n_tokens * top_k
                tpe = max(1, total // num_experts)
                total = tpe * num_experts
                try:
                    token = torch.randn(
                        1, total, K_dim, dtype=torch.float16, device="cuda"
                    )
                    w_raw = torch.randn(
                        num_experts, N_dim, K_dim, dtype=torch.float16, device="cuda"
                    )
                    weight = w_raw.transpose(1, 2)
                    fto = (
                        torch.arange(num_experts, dtype=torch.int32, device="cuda")
                        * tpe
                    ).reshape(-1, 1, 1)

                    cm, _ = benchmark_fn(
                        moe_grouped_matmul,
                        token,
                        weight,
                        fto,
                        mode="none",
                        top_k=top_k,
                        warmup=warmup,
                        repeat=repeat,
                    )

                    def naive(tok, wr, ne, t):
                        out = torch.empty(
                            1,
                            tok.shape[1],
                            wr.shape[1],
                            dtype=tok.dtype,
                            device=tok.device,
                        )
                        for e in range(ne):
                            s = e * t
                            out[0, s : s + t] = tok[0, s : s + t] @ wr[e].T
                        return out

                    nm, _ = benchmark_fn(
                        naive,
                        token,
                        w_raw,
                        num_experts,
                        tpe,
                        warmup=warmup,
                        repeat=repeat,
                    )
                    sp = f"{nm/cm:.2f}x"
                    print(
                        "  ".join(
                            v.ljust(w)
                            for v, w in zip(
                                [
                                    f"{name}/{proj}",
                                    str(n_tokens),
                                    str(num_experts),
                                    str(top_k),
                                    f"{cm:.0f}",
                                    f"{nm:.0f}",
                                    sp,
                                ],
                                W,
                            )
                        )
                    )
                except Exception as e:
                    print(f"  {name}/{proj} {n_tokens}: ERR {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE Grouped Matmul.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    print(f"PyTorch: {torch.__version__}, Device: {torch.cuda.get_device_name()}")
    bench_moe(args.warmup, args.repeat)
    print("\nDone.")


if __name__ == "__main__":
    main()
