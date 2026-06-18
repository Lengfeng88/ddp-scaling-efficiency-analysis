"""
2-GPU DistributedDataParallel (DDP) benchmark for ResNet50.

Launch with torchrun, e.g.:
    torchrun --nproc_per_node=2 ddp_benchmark.py --fp16

Each rank trains on its own batch_size-sized shard; the effective global
batch size is batch_size * world_size. Gradients are synchronized via
NCCL AllReduce inside DDP's backward hooks (see README for how this works).
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP


def main(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model = models.resnet50().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=args.bucket_cap_mb)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None

    dummy_input = torch.randn(args.batch_size, 3, 224, 224).cuda(rank)
    dummy_label = torch.randint(0, 1000, (args.batch_size,)).cuda(rank)

    def train_step():
        optimizer.zero_grad()
        if args.fp16:
            with torch.amp.autocast("cuda"):
                loss = criterion(ddp_model(dummy_input), dummy_label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(ddp_model(dummy_input), dummy_label)
            loss.backward()
            optimizer.step()

    for _ in range(5):
        train_step()

    torch.cuda.synchronize(rank)
    start = time.time()
    for _ in range(args.num_steps):
        train_step()
    torch.cuda.synchronize(rank)
    elapsed = time.time() - start

    if rank == 0:
        world_size = dist.get_world_size()
        throughput = args.num_steps * args.batch_size * world_size / elapsed
        precision = "FP16" if args.fp16 else "FP32"
        print(f"[DDP {world_size}-GPU {precision}] {args.num_steps} steps in {elapsed:.2f}s")
        print(f"[DDP {world_size}-GPU {precision}] Throughput: {throughput:.1f} samples/sec")
        print(f"DDP_THROUGHPUT:{throughput:.1f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64,
                         help="per-GPU batch size")
    parser.add_argument("--bucket_cap_mb", type=int, default=25,
                         help="DDP gradient bucket size in MB")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
