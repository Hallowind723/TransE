from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import FB15kData
from model import TransE


def train_epoch(
    model: TransE,
    data: FB15kData,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    margin: float,
    device: torch.device,
    seed: int,
    epoch: int,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_positive_distance = 0.0
    total_negative_distance = 0.0
    total_triples = 0
    started_at = time.perf_counter()

    batches = data.train_batches(batch_size=batch_size, seed=seed + epoch)
    for batch_number, (positive, negative) in enumerate(batches, start=1):
        positive = positive.to(device)
        negative = negative.to(device)
        current_batch_size = len(positive)

        model.normalize_entity_embeddings_()
        optimizer.zero_grad(set_to_none=True)
        positive_distance = model(positive)
        negative_distance = model(negative)
        loss = F.relu(
            margin + positive_distance - negative_distance
        ).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * current_batch_size
        total_positive_distance += (
            positive_distance.detach().sum().item()
        )
        total_negative_distance += (
            negative_distance.detach().sum().item()
        )
        total_triples += current_batch_size

        if log_interval > 0 and batch_number % log_interval == 0:
            print(
                f"Epoch {epoch:04d} | batch {batch_number:04d} | "
                f"loss {total_loss / total_triples:.4f} | "
                f"positive {total_positive_distance / total_triples:.4f} | "
                f"negative {total_negative_distance / total_triples:.4f}"
            )

    elapsed = time.perf_counter() - started_at
    return {
        "loss": total_loss / total_triples,
        "positive_distance": total_positive_distance / total_triples,
        "negative_distance": total_negative_distance / total_triples,
        "elapsed_seconds": elapsed,
        "num_triples": total_triples,
    }


def save_checkpoint(
    path: Path,
    model: TransE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "model_config": {
                "num_entities": model.num_entities,
                "num_relations": model.num_relations,
                "embedding_dim": model.embedding_dim,
                "distance": model.distance,
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": {
                "data_dir": str(args.data_dir.resolve()),
                "embedding_dim": args.embedding_dim,
                "distance": args.distance,
                "learning_rate": args.learning_rate,
                "margin": args.margin,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            "metrics": metrics,
        },
        path,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train TransE on FB15k")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data" / "FB15k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "runs" / "TransE",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l1")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.margin <= 0:
        parser.error("--margin must be positive")
    if args.log_interval < 0:
        parser.error("--log-interval cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    data = FB15kData.load(args.data_dir)
    model = TransE(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        embedding_dim=args.embedding_dim,
        distance=args.distance,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Training triples: {len(data.train_triples):,}")
    print(
        f"Embedding dim: {args.embedding_dim} | distance: {args.distance} | "
        f"margin: {args.margin} | learning rate: {args.learning_rate}"
    )

    checkpoint_path = args.output_dir / "last.pt"
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model=model,
            data=data,
            optimizer=optimizer,
            batch_size=args.batch_size,
            margin=args.margin,
            device=device,
            seed=args.seed,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        model.normalize_entity_embeddings_()
        save_checkpoint(
            checkpoint_path, model, optimizer, epoch, metrics, args
        )
        print(
            f"Epoch {epoch:04d} complete | "
            f"loss {metrics['loss']:.4f} | "
            f"positive {metrics['positive_distance']:.4f} | "
            f"negative {metrics['negative_distance']:.4f} | "
            f"{metrics['elapsed_seconds']:.1f}s"
        )
        print(f"Checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
