from typing import Callable, Type

from embed_kit.models.adapters.base import EmbeddingAdapter
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.models.adapters")


_adapter_registry: dict[str, type[EmbeddingAdapter]] = {}


def register_adapter(name: str) -> Callable[[Type[EmbeddingAdapter]], Type[EmbeddingAdapter]]:
    def decorator(cls: Type[EmbeddingAdapter]) -> Type[EmbeddingAdapter]:
        _adapter_registry[name.lower()] = cls
        logger.debug(f"Registered adapter: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_adapter_registry() -> dict[str, type["EmbeddingAdapter"]]:
    return _adapter_registry
