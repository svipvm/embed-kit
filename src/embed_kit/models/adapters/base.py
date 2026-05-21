from abc import ABC, abstractmethod
from typing import Any

from embed_kit.models.handler.schemas import (
    DenseEmbeddingOutput,
    MultiEmbeddingOutput,
    SparseEmbedding,
)
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.models.adapters")


class EmbeddingAdapter(ABC):
    def __init__(self, model_name: str, model_path: str, **config: Any) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self._model = None
        self._initialized = False
        
        logger.debug(f"Adapter: {self.__class__.__name__} for {model_name}")

    async def initialize(self) -> None:
        if self._initialized:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        await self._load_model()
        self._initialized = True
        logger.info(f"Model loaded: {self.model_name}")

    @abstractmethod
    async def _load_model(self) -> None:
        pass

    async def encode_dense(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
    ) -> DenseEmbeddingOutput:
        if not self._initialized:
            raise RuntimeError(f"Model not initialized: {self.model_name}")
        
        embeddings = await self._encode_dense_impl(texts, batch_size, max_length)
        
        return DenseEmbeddingOutput(
            embeddings=embeddings,
            model=self.model_name,
            dimensions=len(embeddings[0]) if embeddings else 0,
            num_texts=len(embeddings),
        )

    @abstractmethod
    async def _encode_dense_impl(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        pass

    @abstractmethod
    def count_tokens(self, texts: list[str]) -> int:
        pass

    async def encode_multi(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        return_sparse: bool = True,
    ) -> MultiEmbeddingOutput:
        if not self._initialized:
            raise RuntimeError(f"Model not initialized: {self.model_name}")
        
        return await self._encode_multi_impl(
            texts, batch_size, max_length, return_sparse
        )

    async def _encode_multi_impl(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        return_sparse: bool,
    ) -> MultiEmbeddingOutput:
        if return_sparse:
            logger.warning(
                f"{self.model_name} does not support sparse embeddings. "
                "Falling back to dense-only encoding."
            )
        
        dense_output = await self.encode_dense(texts, batch_size, max_length)
        
        return MultiEmbeddingOutput(
            dense_embeddings=dense_output.embeddings,
            sparse_embeddings=None,
            model=self.model_name,
            dimensions=dense_output.dimensions,
            num_texts=dense_output.num_texts,
        )

    async def shutdown(self) -> None:
        self._model = None
        self._initialized = False
        logger.info(f"Model shutdown: {self.model_name}")
