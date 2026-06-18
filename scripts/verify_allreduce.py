"""
Minimal sanity check for NCCL AllReduce: each rank starts with a different
tensor, calls all_reduce(SUM), and both ranks should end up with identical
averaged values. This is the mechanism DDP relies on to keep model replicas
in sync across GPUs after every backward pass.

Launch with:
    torchrun --nproc_per_node=2 verify_allreduce.py
"""

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda(rank)
    else:
        tensor = torch.tensor([5.0, 6.0, 7.0, 8.0]).cuda(rank)

    print(f"Before AllReduce - rank {rank}: {tensor.tolist()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()

    print(f"After AllReduce  - rank {rank}: {tensor.tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
