from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import jieba
import mmh3
from qdrant_client import models

ALNUM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


@dataclass
class ChineseSparseEncoder:
    """Generate hashed Chinese BM25-style sparse vectors for Qdrant."""

    average_document_length: float = 1.0
    k1: float = 1.2
    b: float = 0.75

    def tokenize(self, text: str) -> list[str]:
        tokens = [token.lower() for token in ALNUM_PATTERN.findall(text)]
        for segment in CHINESE_PATTERN.findall(text):
            tokens.extend(token.strip() for token in jieba.cut_for_search(segment) if token.strip())
        return tokens

    def fit(self, documents: Iterable[str]) -> None:
        lengths = [len(self.tokenize(document)) for document in documents]
        self.average_document_length = sum(lengths) / max(len(lengths), 1)
        if self.average_document_length <= 0:
            self.average_document_length = 1.0

    @staticmethod
    def _token_index(token: str) -> int:
        return mmh3.hash(token, signed=False)

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = self.tokenize(text)
        counts = Counter(tokens)
        document_length = max(len(tokens), 1)
        denominator_adjustment = self.k1 * (
            1 - self.b + self.b * document_length / self.average_document_length
        )
        weighted = {
            self._token_index(token): (frequency * (self.k1 + 1))
            / (frequency + denominator_adjustment)
            for token, frequency in counts.items()
        }
        ordered = sorted(weighted.items())
        return models.SparseVector(
            indices=[index for index, _ in ordered],
            values=[value for _, value in ordered],
        )

    def encode_query(self, text: str) -> models.SparseVector:
        indices = sorted({self._token_index(token) for token in self.tokenize(text)})
        return models.SparseVector(indices=indices, values=[1.0] * len(indices))
