from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMedicalModel(ABC):
    """Minimal model interface for training/evaluation/serving parity."""

    @abstractmethod
    def predict(self, message: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def train(self, **kwargs: Any) -> None:
        raise NotImplementedError('This RAG module builds KG/vector indexes instead of gradient training.')

    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        results = [self.predict(sample['question']) for sample in samples]
        return {'num_samples': len(samples), 'results': results}
