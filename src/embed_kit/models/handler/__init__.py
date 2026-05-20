from embed_kit.models.handler.base import BaseModelHandler
from embed_kit.models.handler.schemas import (
    DenseEmbeddingOutput,
    EmbeddingInput,
    MultiEmbeddingOutput,
    SparseEmbedding,
)

__all__ = [
    "BaseModelHandler",
    "EmbeddingModelHandler",
    "create_model_handler",
    "EmbeddingInput",
    "DenseEmbeddingOutput",
    "MultiEmbeddingOutput",
    "SparseEmbedding",
]


def __getattr__(name: str):
    if name == "EmbeddingModelHandler":
        from embed_kit.models.handler.embedding import EmbeddingModelHandler
        return EmbeddingModelHandler
    elif name == "create_model_handler":
        from embed_kit.models.handler.embedding import create_model_handler
        return create_model_handler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
