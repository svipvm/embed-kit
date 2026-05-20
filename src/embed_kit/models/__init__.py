from embed_kit.models.adapters import BGEM3Adapter, EmbeddingAdapter, get_adapter_registry, register_adapter
from embed_kit.models.handler import (
    BaseModelHandler,
    EmbeddingModelHandler,
    create_model_handler,
    DenseEmbeddingOutput,
    EmbeddingInput,
    MultiEmbeddingOutput,
    SparseEmbedding,
)

__all__ = [
    "BaseModelHandler",
    "EmbeddingModelHandler",
    "create_model_handler",
    "EmbeddingAdapter",
    "BGEM3Adapter",
    "register_adapter",
    "get_adapter_registry",
    "EmbeddingInput",
    "DenseEmbeddingOutput",
    "MultiEmbeddingOutput",
    "SparseEmbedding",
]
