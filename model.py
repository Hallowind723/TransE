from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TransE(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 50,
        distance: str = "l1",
    ) -> None:
        super().__init__()
        if num_entities <= 0:
            raise ValueError("num_entities must be positive")
        if num_relations <= 0:
            raise ValueError("num_relations must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        distance = distance.lower()
        if distance not in {"l1", "l2"}:
            raise ValueError("distance must be 'l1' or 'l2'")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.norm_p = 1 if distance == "l1" else 2

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        self._normalize_rows_(self.relation_embeddings.weight)

    @torch.no_grad()
    def normalize_entity_embeddings_(self) -> None:
        self._normalize_rows_(self.entity_embeddings.weight)

    @staticmethod
    @torch.no_grad()
    def _normalize_rows_(embeddings: Tensor) -> None:
        norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
        embeddings.div_(norms.clamp_min(torch.finfo(embeddings.dtype).eps))

    def forward(self, triples: Tensor) -> Tensor:
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must have shape [batch_size, 3]")

        heads, relations, tails = triples.unbind(dim=1)
        head_vectors = self.entity_embeddings(heads)
        relation_vectors = self.relation_embeddings(relations)
        tail_vectors = self.entity_embeddings(tails)
        residual = head_vectors + relation_vectors - tail_vectors
        return torch.linalg.vector_norm(residual, ord=self.norm_p, dim=1)
