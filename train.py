from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import FB15kData
from evaluate import evaluate_link_prediction
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
        ).sum()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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
    validation_metrics: dict[str, float] | None = None,
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
                "validation_interval": args.validation_interval,
                "validation_limit": args.validation_limit,
            },
            "metrics": metrics,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    workspace_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train TransE on FB15k")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=workspace_dir / "data" / "FB15k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_dir / "runs" / "TransE",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l1")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=1000,
        help="Number of validation triples per evaluation; 0 uses all triples",
    )
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
    if args.validation_interval <= 0:
        parser.error("--validation-interval must be positive")
    if args.validation_limit < 0:
        parser.error("--validation-limit cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    best_checkpoint_path = args.output_dir / "best.pt"
    best_validation_raw_mr = float("inf")
    validation_triples = (
        data.valid_triples
        if args.validation_limit == 0
        else data.valid_triples[: args.validation_limit]
    )
    filter_triple_sets = (
        data.train_triples,
        data.valid_triples,
        data.test_triples,
    )
    print(
        f"Validation: every {args.validation_interval} epochs | "
        f"triples {len(validation_triples):,}"
    )

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
        print(
            f"Epoch {epoch:04d} complete | "
            f"loss {metrics['loss']:.4f} | "
            f"positive {metrics['positive_distance']:.4f} | "
            f"negative {metrics['negative_distance']:.4f} | "
            f"{metrics['elapsed_seconds']:.1f}s"
        )

        validation_metrics = None
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            validation_metrics = evaluate_link_prediction(
                model=model,
                triples=validation_triples,
                filter_triple_sets=filter_triple_sets,
                log_interval=0,
            )
            print(
                f"Validation {epoch:04d} | "
                f"raw MR {validation_metrics['raw_mean_rank']:.2f} | "
                f"filtered MR {validation_metrics['filtered_mean_rank']:.2f} | "
                f"raw Hits@10 {validation_metrics['raw_hits_at_10']:.2%} | "
                f"filtered Hits@10 "
                f"{validation_metrics['filtered_hits_at_10']:.2%}"
            )

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            metrics,
            args,
            validation_metrics,
        )
        print(f"Checkpoint: {checkpoint_path.resolve()}")

        if (
            validation_metrics is not None
            and validation_metrics["raw_mean_rank"] < best_validation_raw_mr
        ):
            best_validation_raw_mr = validation_metrics["raw_mean_rank"]
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                metrics,
                args,
                validation_metrics,
            )
            print(
                f"Best checkpoint: {best_checkpoint_path.resolve()} | "
                f"raw MR {best_validation_raw_mr:.2f}"
            )


if __name__ == "__main__":
    main()
