import time, torch
torch.manual_seed(0)

def run(m, nheads, seqlen, dhead, warmup=10, iters=50, device="cuda"):
    q = torch.randn(m, nheads, seqlen, dhead, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    torch.cuda.synchronize()
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available"); exit(0)
    dt = run(m=1, nheads=24, seqlen=4096, dhead=64)
    print({"sdpa_fp16_avg_s": dt})
