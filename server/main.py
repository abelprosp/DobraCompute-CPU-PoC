
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import time

import importlib
try:
    dobra_qgemm = importlib.import_module("dobra_qgemm")
except Exception as e:
    dobra_qgemm = None
    print("WARNING: dobra_qgemm not found. Build the extension first. Error:", e)

app = FastAPI(title="DobraCompute CPU PoC", version="0.1.0")

class BenchmarkRequest(BaseModel):
    M: int = 1024
    K: int = 1024
    N: int = 1024
    iters: int = 5
    dequant: bool = True

class InferRequest(BaseModel):
    # Placeholder: future will accept token ids and KV-cache handles
    batch: int = 1
    seq_len: int = 128
    dim: int = 1024

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    assert dobra_qgemm is not None, "Build dobra_qgemm first"
    M, K, N = req.M, req.K, req.N
    A = np.random.randint(-127, 127, size=(M,K), dtype=np.int8)
    B = np.random.randint(-127, 127, size=(K,N), dtype=np.int8)
    sa = np.ones((M,), dtype=np.float32) * (1.0/127.0)
    sb = np.ones((N,), dtype=np.float32) * (1.0/127.0)
    # Warmup
    dobra_qgemm.qgemm_i8i8(A, B, sa, sb, req.dequant)
    times = []
    ops = 2.0 * M * K * N  # int8 MACs ~ 2 ops
    for _ in range(req.iters):
        t0 = time.perf_counter()
        C32, Cf = dobra_qgemm.qgemm_i8i8(A, B, sa, sb, req.dequant)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    best = min(times)
    tflops = ops / best / 1e12
    return {
        "M": M, "K": K, "N": N,
        "iters": req.iters,
        "best_s": best,
        "avg_s": sum(times)/len(times),
        "approx_ops": ops,
        "approx_TFLOPS": tflops
    }

@app.post("/infer")
def infer(req: InferRequest):
    # Placeholder for pipeline stages; here we just exercise GEMM as MLP proxy
    assert dobra_qgemm is not None, "Build dobra_qgemm first"
    B, T, D = req.batch, req.seq_len, req.dim
    A = np.random.randint(-127, 127, size=(B*T,D), dtype=np.int8)
    W = np.random.randint(-127, 127, size=(D,D), dtype=np.int8)
    sa = np.ones((B*T,), dtype=np.float32) * (1.0/127.0)
    sb = np.ones((D,), dtype=np.float32) * (1.0/127.0)
    C32, Cf = dobra_qgemm.qgemm_i8i8(A, W, sa, sb, True)
    # return only summaries to avoid huge payload
    return {"ok": True, "out_shape": [B*T, D], "sum": float(np.sum(Cf)), "mean": float(np.mean(Cf))}
