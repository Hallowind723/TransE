from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from dataset import FB15kData
from model import TransE
from train import resolve_device


FilterIndex = dict[tuple[int, int], tuple[int, ...]]


def build_filter_index(
    triple_sets: Iterable[Tensor],
) -> tuple[FilterIndex, FilterIndex]:
    head_candidates: dict[tuple[int, int], list[int]] = {}
    tail_candidates: dict[tuple[int, int], list[int]] = {}

    for triples in triple_sets:
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("filter triples must have shape [num_triples, 3]")
        for head, relation, tail in triples.tolist():
            head_candidates.setdefault((relation, tail), []).append(head)
            tail_candidates.setdefault((head, relation), []).append(tail)

    head_filters = {
        key: tuple(set(candidates)) for key, candidates in head_candidates.items()
    }
    tail_filters = {
        key: tuple(set(candidates)) for key, candidates in tail_candidates.items()
    }
    return head_filters, tail_filters


def _candidate_distances(
    model: TransE,
    triples: Tensor,
    start: int,
    end: int,
    side: str,
) -> Tensor:
    device = triples.device
    candidate_ids = torch.arange(start, end, device=device)
    candidate_vectors = model.entity_embeddings(candidate_ids)
    relation_vectors = model.relation_embeddings(triples[:, 1])

    if side == "tail":
        head_vectors = model.entity_embeddings(triples[:, 0])
        residual = (
            head_vectors[:, None, :]
            + relation_vectors[:, None, :]
            - candidate_vectors[None, :, :]
        )
    else:
        tail_vectors = model.entity_embeddings(triples[:, 2])
        residual = (
            candidate_vectors[None, :, :]
            + relation_vectors[:, None, :]
            - tail_vectors[:, None, :]
        )

    return torch.linalg.vector_norm(residual, ord=model.norm_p, dim=2)


@torch.inference_mode()
def evaluate_link_prediction(
    model: TransE,
    triples: Tensor,
    filter_triple_sets: Iterable[Tensor],
    triple_batch_size: int = 32,
    candidate_batch_size: int = 1024,
    log_interval: int = 1000,
) -> dict[str, float]:
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triples must have shape [num_triples, 3]")
    if len(triples) == 0:
        raise ValueError("triples cannot be empty")
    if triple_batch_size <= 0:
        raise ValueError("triple_batch_size must be positive")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if log_interval < 0:
        raise ValueError("log_interval cannot be negative")

    head_filters, tail_filters = build_filter_index(filter_triple_sets)
    device = model.entity_embeddings.weight.device
    was_training = model.training
    model.eval()

    rank_sums = {
        f"{side}_{setting}": 0
        for side in ("head", "tail")
        for setting in ("raw", "filtered")
    }
    hits = dict.fromkeys(rank_sums, 0)
    next_log = log_interval

    try:
        for start in range(0, len(triples), triple_batch_size):
            batch = triples[start : start + triple_batch_size].to(device)
            batch_list = batch.cpu().tolist()
            batch_size = len(batch)
            target_distances = model(batch)

            for side in ("head", "tail"):
                raw_ranks = torch.ones(batch_size, dtype=torch.long, device=device)
                filtered_ranks = torch.ones_like(raw_ranks)

                if side == "head":
                    filters = head_filters
                    filter_keys = [(r, t) for _, r, t in batch_list]
                    targets = [h for h, _, _ in batch_list]
                else:
                    filters = tail_filters
                    filter_keys = [(h, r) for h, r, _ in batch_list]
                    targets = [t for _, _, t in batch_list]

                for candidate_start in range(
                    0, model.num_entities, candidate_batch_size
                ):
                    candidate_end = min(
                        candidate_start + candidate_batch_size,
                        model.num_entities,
                    )
                    distances = _candidate_distances(
                        model, batch, candidate_start, candidate_end, side
                    )
                    closer = distances < target_distances[:, None]
                    raw_ranks += closer.sum(dim=1)

                    filter_mask = torch.zeros_like(closer)
                    for row, (key, target) in enumerate(
                        zip(filter_keys, targets, strict=True)
                    ):
                        for entity in filters.get(key, ()):
                            if candidate_start <= entity < candidate_end:
                                filter_mask[row, entity - candidate_start] = True
                        if candidate_start <= target < candidate_end:
                            filter_mask[row, target - candidate_start] = False
                    filtered_ranks += (closer & ~filter_mask).sum(dim=1)

                for setting, batch_ranks in (
                    ("raw", raw_ranks),
                    ("filtered", filtered_ranks),
                ):
                    name = f"{side}_{setting}"
                    rank_sums[name] += batch_ranks.sum().item()
                    hits[name] += (batch_ranks <= 10).sum().item()

            processed = min(start + batch_size, len(triples))
            if log_interval > 0 and processed >= next_log:
                print(f"Evaluated {processed:,}/{len(triples):,} triples")
                while next_log <= processed:
                    next_log += log_interval
    finally:
        if was_training:
            model.train()

    metrics: dict[str, float] = {}
    for side in ("head", "tail"):
        for setting in ("raw", "filtered"):
            name = f"{side}_{setting}"
            metrics[f"{name}_mean_rank"] = rank_sums[name] / len(triples)
            metrics[f"{name}_hits_at_10"] = hits[name] / len(triples)

    prediction_count = 2 * len(triples)
    for setting in ("raw", "filtered"):
        rank_sum = rank_sums[f"head_{setting}"] + rank_sums[f"tail_{setting}"]
        hit_count = hits[f"head_{setting}"] + hits[f"tail_{setting}"]
        metrics[f"{setting}_mean_rank"] = rank_sum / prediction_count
        metrics[f"{setting}_hits_at_10"] = hit_count / prediction_count

    return metrics


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate TransE link prediction")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data" / "FB15k",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "runs" / "TransE" / "last.pt",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--triple-batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.triple_batch_size <= 0:
        parser.error("--triple-batch-size must be positive")
    if args.candidate_batch_size <= 0:
        parser.error("--candidate-batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.log_interval < 0:
        parser.error("--log-interval cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    data = FB15kData.load(args.data_dir)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    model = TransE(**checkpoint["model_config"])
    if (
        model.num_entities != data.num_entities
        or model.num_relations != data.num_relations
    ):
        raise ValueError("Checkpoint and dataset entity/relation counts do not match")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    triples = data.valid_triples if args.split == "valid" else data.test_triples
    if args.limit is not None:
        triples = triples[: args.limit]

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Split: {args.split} | Triples: {len(triples):,}")
    if args.limit is not None:
        print("The --limit option is for smoke checks, not paper-comparable results")

    metrics = evaluate_link_prediction(
        model=model,
        triples=triples,
        filter_triple_sets=(
            data.train_triples,
            data.valid_triples,
            data.test_triples,
        ),
        triple_batch_size=args.triple_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        log_interval=args.log_interval,
    )

    print("\nCombined head and tail prediction")
    print(f"Raw MR:       {metrics['raw_mean_rank']:.2f}")
    print(f"Filtered MR:  {metrics['filtered_mean_rank']:.2f}")
    print(f"Raw Hits@10:  {metrics['raw_hits_at_10']:.2%}")
    print(f"Filt Hits@10: {metrics['filtered_hits_at_10']:.2%}")
    for side in ("head", "tail"):
        print(f"\n{side.capitalize()} prediction")
        print(
            f"Raw MR: {metrics[f'{side}_raw_mean_rank']:.2f} | "
            f"Filtered MR: {metrics[f'{side}_filtered_mean_rank']:.2f}"
        )
        print(
            f"Raw Hits@10: {metrics[f'{side}_raw_hits_at_10']:.2%} | "
            f"Filtered Hits@10: {metrics[f'{side}_filtered_hits_at_10']:.2%}"
        )


if __name__ == "__main__":
    main()
