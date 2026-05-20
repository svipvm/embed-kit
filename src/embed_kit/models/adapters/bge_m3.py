from typing import Any

from embed_kit.models.adapters.base import EmbeddingAdapter
from embed_kit.models.adapters.registry import register_adapter
from embed_kit.models.handler.schemas import MultiEmbeddingOutput, SparseEmbedding
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.models.adapters.bge_m3")


@register_adapter("bge-m3")
@register_adapter("baai/bge-m3")
class BGEM3Adapter(EmbeddingAdapter):
    async def _load_model(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        
        use_fp16 = self.config.get("use_fp16", True)
        device = self.config.get("device", "cuda")
        normalize_embeddings = self.config.get("normalize_embeddings", True)
        
        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=use_fp16,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )
        logger.info(f"BGE-M3 loaded from {self.model_path}")

    async def _encode_dense_impl(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        if not texts:
            return []
        
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        
        return result["dense_vecs"].tolist()

    async def _encode_multi_impl(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        return_sparse: bool,
    ) -> MultiEmbeddingOutput:
        if not texts:
            return MultiEmbeddingOutput(
                dense_embeddings=[],
                sparse_embeddings=None,
                model=self.model_name,
                dimensions=0,
                num_texts=0,
            )
        
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        
        dense_embeddings = result["dense_vecs"].tolist()
        dimensions = len(dense_embeddings[0]) if dense_embeddings else 0
        
        sparse_embeddings = None
        if return_sparse and "lexical_weights" in result:
            sparse_embeddings = []
            for sparse_dict in result["lexical_weights"]:
                indices = list(sparse_dict.keys())
                values = [float(v) for v in sparse_dict.values()]
                sparse_embeddings.append(SparseEmbedding(indices=indices, values=values))
        
        return MultiEmbeddingOutput(
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            model=self.model_name,
            dimensions=dimensions,
            num_texts=len(dense_embeddings),
        )
