from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class FB15kData:
    data_dir: Path
    entity2id: dict[str, int]
    relation2id: dict[str, int]
    train_triples: Tensor
    valid_triples: Tensor
    test_triples: Tensor

    @classmethod
    def load(cls, data_dir: str | Path) -> FB15kData:
        data_dir = Path(data_dir)
        entity2id = _read_mapping(data_dir / "entity2id.txt")
        relation2id = _read_mapping(data_dir / "relation2id.txt")

        return cls(
            data_dir=data_dir,
            entity2id=entity2id,
            relation2id=relation2id,
            train_triples=_read_triples(
                data_dir / "train.txt", entity2id, relation2id
            ),
            valid_triples=_read_triples(
                data_dir / "valid.txt", entity2id, relation2id
            ),
            test_triples=_read_triples(
                data_dir / "test.txt", entity2id, relation2id
            ),
        )

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    @cached_property
    def _train_triple_set(self) -> set[tuple[int, int, int]]:
        return {tuple(triple) for triple in self.train_triples.tolist()}

    def corrupt_triples(
        self, positive: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        if positive.ndim != 2 or positive.shape[1] != 3:
            raise ValueError("positive must have shape [batch_size, 3]")

        batch_size = positive.shape[0]
        negative = positive.clone()
        replace_head = (
            torch.rand(batch_size, generator=generator) < 0.5
        )
        random_entities = torch.randint(
            self.num_entities, (batch_size,), generator=generator
        )
        negative[replace_head, 0] = random_entities[replace_head]
        negative[~replace_head, 2] = random_entities[~replace_head]

        for index in range(batch_size):
            while tuple(negative[index].tolist()) in self._train_triple_set:
                replacement = torch.randint(
                    self.num_entities, (1,), generator=generator
                ).item()
                column = 0 if replace_head[index] else 2
                negative[index, column] = replacement

        return negative

    def train_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        generator = torch.Generator()
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        indices = (
            torch.randperm(len(self.train_triples), generator=generator)
            if shuffle
            else torch.arange(len(self.train_triples))
        )
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            positive = self.train_triples[batch_indices]
            negative = self.corrupt_triples(positive, generator)
            yield positive, negative


def _read_mapping(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    used_ids: set[int] = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected name and integer ID"
                )
            name, raw_id = parts
            try:
                identifier = int(raw_id)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid integer ID {raw_id!r}"
                ) from error
            if name in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate name {name!r}")
            if identifier in used_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate ID {identifier}"
                )
            mapping[name] = identifier
            used_ids.add(identifier)

    if used_ids != set(range(len(mapping))):
        raise ValueError(f"{path}: IDs must be contiguous and start at 0")
    return mapping


def _read_triples(
    path: Path,
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> Tensor:
    triples: list[tuple[int, int, int]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected head, tail, and relation"
                )
            head, tail, relation = parts
            try:
                triples.append(
                    (
                        entity2id[head],
                        relation2id[relation],
                        entity2id[tail],
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"{path}:{line_number}: unknown identifier {error.args[0]!r}"
                ) from error

    return torch.tensor(triples, dtype=torch.long)


def main() -> None:
    default_data_dir = Path(__file__).resolve().parents[2] / "data" / "FB15k"
    parser = argparse.ArgumentParser(description="Load and inspect FB15k")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = FB15kData.load(args.data_dir)
    print(f"Data directory: {data.data_dir.resolve()}")
    print(f"Entities: {data.num_entities:,}")
    print(f"Relations: {data.num_relations:,}")
    print(f"Train: {data.train_triples.shape}")
    print(f"Valid: {data.valid_triples.shape}")
    print(f"Test: {data.test_triples.shape}")

    positive, negative = next(
        data.train_batches(batch_size=args.batch_size, seed=args.seed)
    )
    print(f"Positive batch: {positive.shape}, dtype={positive.dtype}")
    print(f"Negative batch: {negative.shape}, dtype={negative.dtype}")
    print(f"First positive (h, r, t): {positive[0].tolist()}")
    print(f"First negative (h, r, t): {negative[0].tolist()}")


if __name__ == "__main__":
    main()
