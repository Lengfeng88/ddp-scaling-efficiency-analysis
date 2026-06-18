"""
Sweep bucket_cap_mb from 1MB to 100MB and report throughput + scaling
efficiency for each setting. Helps find the optimal gradient bucket size
for a given model and GPU interconnect.

Launch with:
    torchrun --nproc_per_node=2 bucket_sweep.py
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

SINGLE_GPU_BASELINE = 100.6  # samples/sec — from single_gpu_baseline.py
BUCKET_SIZES = [1, 5, 10, 25, 50, 100]


def benchmark(bucket_mb: int, num_steps: int = 30) -> float:
    rank = dist.get_rank()
    model = models.resnet50().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=bucket_mb)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(64, 3, 224, 224).cuda(rank)
    y = torch.randint(0, 1000, (64,)).cuda(rank)

    for _ in range(5):
        optimizer.zero_grad()
        criterion(ddp_model(x), y).backward()
        optimizer.step()

    torch.cuda.synchronize(rank)
    t0 = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        criterion(ddp_model(x), y).backward()
        optimizer.step()
    torch.cuda.synchronize(rank)
    elapsed = time.time() - t0

    tp = num_steps * 64 * dist.get_world_size() / elapsed
    del ddp_model, model, optimizer
    torch.cuda.empty_cache()
    return tp


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    results = {}
    for bucket_mb in BUCKET_SIZES:
        tp = benchmark(bucket_mb)
        results[bucket_mb] = tp
        if rank == 0:
            eff = (tp / dist.get_world_size()) / SINGLE_GPU_BASELINE * 100
            print(f"bucket={bucket_mb:>4}MB  throughput={tp:>7.1f} s/s  "
                  f"scaling_eff={eff:.1f}%")

    if rank == 0:
        best = max(results, key=results.get)
        best_eff = (results[best] / dist.get_world_size()) / SINGLE_GPU_BASELINE * 100
        print(f"\nBest bucket size: {best}MB  "
              f"(throughput={results[best]:.1f}, efficiency={best_eff:.1f}%)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
