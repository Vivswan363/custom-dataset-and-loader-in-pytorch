import argparse
import importlib
import multiprocessing as mp
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BufferedBatchedDataset, BufferedDataset


def load_model(model_name: str, input_dim: int, output_dim: int):
    module = importlib.import_module(f"models.{model_name}")

    if not hasattr(module, "build_model"):
        raise AttributeError(
            f"models/{model_name}.py must define build_model(input_dim, output_dim)"
        )

    return module.build_model(input_dim=input_dim, output_dim=output_dim)


import time
import torch
import torch.nn as nn


def train_one_epoch(
    model,
    loader,
    device,
    lr=1e-3,
    log_every=10_000,
):
    model.to(device)
    model.train()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    total_dataload_time = 0.0
    total_train_time = 0.0

    epoch_start = time.perf_counter()

    loader_iter = iter(loader)
    batch_idx = 0

    while True:
        # -------------------------
        # Dataloading time
        # -------------------------
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()

        total_dataload_time += t1 - t0
        batch_idx += 1

        # -------------------------
        # Training time
        # -------------------------
        t0 = time.perf_counter()

        batch = batch.to(device, non_blocking=True).float()

        x = batch[:, 1:]              # columns 1..79
        y = batch[:, 0].unsqueeze(1)  # column 0

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



        t1 = time.perf_counter()

        total_train_time += t1 - t0

        bs = batch.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
        total_batches += 1

        if batch_idx % log_every == 0:
            elapsed = time.perf_counter() - epoch_start

            print(
                f"batch={batch_idx} "
                f"loss={loss.item():.6f} "
                f"samples/sec={total_samples / elapsed:,.0f} "
                f"dataload_time={total_dataload_time:.2f}s "
                f"train_time={total_train_time:.2f}s"
            )

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    return {
        "batches": total_batches,
        "samples": total_samples,
        "avg_loss": total_loss / total_samples,
        "epoch_time_sec": epoch_time,
        "samples_per_sec": total_samples / epoch_time,
        "dataload_time_sec": total_dataload_time,
        "train_time_sec": total_train_time,
        "other_time_sec": epoch_time - total_dataload_time - total_train_time,
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="linear_model")
    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BufferedBatchedDataset(
        data_dir=args.data_dir,
        buffer_size=args.buffer_size,
        max_files=args.max_files,
        batch_size=args.batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,  # dataset already yields batches
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=device == "cuda",
    )

    model = load_model(
        model_name=args.model,
        input_dim=79,
        output_dim=1,
    )

    print(f"Model: {args.model}")
    print(f"Device: {device}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        stats = train_one_epoch(
            model=model,
            loader=loader,
            device=device,
            lr=args.lr,
        )

        print(
            f"epoch={epoch + 1} "
            f"batches={stats['batches']} "
            f"samples={stats['samples']:,} "
            f"avg_loss={stats['avg_loss']:.6f} "
            f"dataload_time={stats['dataload_time_sec']:.2f}s "
            f"train_time={stats['train_time_sec']:.2f}s "
            f"samples/sec={stats['samples_per_sec']:,.0f}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()