"""
PyTorch Profiler-based breakdown of where training time goes in 2-GPU DDP.
Prints the top CUDA-time operations and isolates NCCL/AllReduce time as a
fraction of total CUDA time, which is the basis for the "X% communication
overhead" figure reported in the README.

Launch with:
    torchrun --nproc_per_node=2 profile_ddp.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model = models.resnet50().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dummy_input = torch.randn(64, 3, 224, 224).cuda(rank)
    dummy_label = torch.randint(0, 1000, (64,)).cuda(rank)

    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(ddp_model(dummy_input), dummy_label)
        loss.backward()
        optimizer.step()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=8),
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            optimizer.zero_grad()
            with torch.profiler.record_function("forward"):
                output = ddp_model(dummy_input)
                loss = criterion(output, dummy_label)
            with torch.profiler.record_function("backward"):
                loss.backward()
            with torch.profiler.record_function("optimizer"):
                optimizer.step()
            prof.step()

    if rank == 0:
        print("\n=== Top 20 CUDA operations ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        total_cuda = sum(e.cuda_time_total for e in prof.key_averages())
        comm_cuda = sum(
            e.cuda_time_total
            for e in prof.key_averages()
            if "allreduce" in e.key.lower() or "nccl" in e.key.lower()
        )
        if total_cuda > 0:
            print("\n=== Communication overhead ===")
            print(f"Total CUDA time:      {total_cuda / 1e6:.3f}s")
            print(f"AllReduce CUDA time:  {comm_cuda / 1e6:.3f}s")
            print(f"Communication ratio:  {comm_cuda / total_cuda * 100:.1f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
