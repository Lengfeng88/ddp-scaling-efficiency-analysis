"""
Single-GPU baseline benchmark for ResNet50 training.
Measures throughput (samples/sec) to compare against DDP scaling later.
"""

import time
import torch
import torch.nn as nn
import torchvision.models as models


def run_single_gpu(num_steps=50, use_fp16=False, batch_size=64):
    model = models.resnet50().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dummy_input = torch.randn(batch_size, 3, 224, 224).cuda(0)
    dummy_label = torch.randint(0, 1000, (batch_size,)).cuda(0)

    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    def train_step():
        optimizer.zero_grad()
        if use_fp16:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(dummy_input), dummy_label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(dummy_input), dummy_label)
            loss.backward()
            optimizer.step()

    # Warmup: excludes cuDNN autotuning and lazy kernel compilation from timing
    for _ in range(5):
        train_step()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_steps):
        train_step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = num_steps * batch_size / elapsed
    precision = "FP16" if use_fp16 else "FP32"
    print(f"[Single GPU {precision}] {num_steps} steps in {elapsed:.2f}s")
    print(f"[Single GPU {precision}] Throughput: {throughput:.1f} samples/sec")
    return throughput


if __name__ == "__main__":
    print("=== FP32 ===")
    run_single_gpu(use_fp16=False)
    print("\n=== FP16 ===")
    run_single_gpu(use_fp16=True)
