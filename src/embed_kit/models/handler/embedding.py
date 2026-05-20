from typing import Any

from embed_kit.models.adapters.base import EmbeddingAdapter
from embed_kit.models.adapters.bge_m3 import BGEM3Adapter
from embed_kit.models.adapters.registry import get_adapter_registry
from embed_kit.models.handler.base import BaseModelHandler
from embed_kit.models.handler.schemas import (
    DenseEmbeddingOutput,
    EmbeddingInput,
    MultiEmbeddingOutput,
)
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.models.handler")


def create_adapter(model_id: str, model_name: str, model_path: str, **config: Any) -> EmbeddingAdapter:
    adapter_registry = get_adapter_registry()
    adapter_name = config.get("adapter")
    
    adapter_cls = (
        (adapter_registry.get(adapter_name.lower()) if adapter_name else None) or
        adapter_registry.get(model_id.lower()) or
        adapter_registry.get(model_name.lower()) or
        BGEM3Adapter
    )
    
    logger.info(f"Creating adapter: {adapter_cls.__name__} for {model_name}")
    return adapter_cls(model_name=model_name, model_path=model_path, **config)


class EmbeddingModelHandler(BaseModelHandler[EmbeddingInput, DenseEmbeddingOutput]):
    def __init__(self, model_id: str = "unknown", **config: Any) -> None:
        super().__init__(**config)
        
        self.model_id = model_id
        model_name = config.get("model_name", "unknown")
        model_path = config.get("model_path", "")
        
        if not model_path:
            raise ValueError("model_path is required")
        
        adapter_config = {k: v for k, v in config.items() 
                         if k not in ["model_name", "model_path"]}
        
        self._adapter = create_adapter(
            model_id=model_id,
            model_name=model_name,
            model_path=model_path,
            **adapter_config,
        )

    @property
    def input_schema(self) -> type[EmbeddingInput]:
        return EmbeddingInput

    @property
    def output_schema(self) -> type[DenseEmbeddingOutput]:
        return DenseEmbeddingOutput

    async def _setup(self) -> None:
        await self._adapter.initialize()

    async def process(self, input_data: EmbeddingInput) -> DenseEmbeddingOutput:
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        return await self._adapter.encode_dense(
            texts=input_data.texts,
            batch_size=input_data.batch_size,
            max_length=input_data.max_length,
        )

    async def encode_multi(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        return_sparse: bool = True,
    ) -> MultiEmbeddingOutput:
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        return await self._adapter.encode_multi(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            return_sparse=return_sparse,
        )

    async def shutdown(self) -> None:
        await self._adapter.shutdown()
        await super().shutdown()
    
    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(text.split()) for text in texts)


def create_model_handler(model_id: str, **config: Any) -> EmbeddingModelHandler:
    return EmbeddingModelHandler(model_id=model_id, **config)
