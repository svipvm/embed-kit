from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from embed_kit.api.schemas import (
    EmbeddingData,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    SparseEmbedding,
    UsageInfo,
)
from embed_kit.models.handler.base import BaseModelHandler
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.api.routes")

router = APIRouter()

_model_handler: BaseModelHandler | None = None


def get_model_handler() -> BaseModelHandler:
    global _model_handler
    if _model_handler is None:
        logger.error("Model handler not initialized")
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _model_handler


def set_model_handler(handler: BaseModelHandler) -> None:
    global _model_handler
    _model_handler = handler
    logger.debug(f"Model handler set: {handler.__class__.__name__}")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    handler: BaseModelHandler = Depends(get_model_handler),
) -> EmbeddingResponse:
    try:
        texts = [request.input] if isinstance(request.input, str) else request.input
        logger.info(f"Creating embeddings for {len(texts)} texts, model={request.model}")

        from embed_kit.models.handler import EmbeddingInput

        embedding_input = EmbeddingInput(
            texts=texts,
            model=request.model,
        )
        result = await handler.process(embedding_input)

        data = [
            EmbeddingObject(
                object="embedding",
                embedding=EmbeddingData(
                    dense=emb,
                    sparse=SparseEmbedding(),
                ),
                index=i,
            )
            for i, emb in enumerate(result.embeddings)
        ]

        total_tokens = handler.count_tokens(texts)
        logger.debug(f"Embedding complete: dimensions={result.dimensions}, tokens={total_tokens}")

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed_all", response_model=EmbeddingResponse)
async def create_embeddings_all(
    request: EmbeddingRequest,
    handler: BaseModelHandler = Depends(get_model_handler),
) -> EmbeddingResponse:
    try:
        texts = [request.input] if isinstance(request.input, str) else request.input
        logger.info(f"Creating embeddings for {len(texts)} texts, model={request.model}")

        batch_size = handler.config["batch_size"]
        max_length = handler.config["max_length"]

        result = await handler.encode_multi(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            return_sparse=True,
        )

        data = []
        for i, dense_emb in enumerate(result.dense_embeddings):
            sparse_emb = SparseEmbedding()
            if result.sparse_embeddings and i < len(result.sparse_embeddings):
                sparse_obj = result.sparse_embeddings[i]
                sparse_emb = SparseEmbedding(
                    indices=sparse_obj.indices,
                    values=sparse_obj.values,
                )

            embedding_data = EmbeddingData(
                dense=dense_emb,
                sparse=sparse_emb,
            )

            data.append(
                EmbeddingObject(
                    object="embedding",
                    embedding=embedding_data,
                    index=i,
                )
            )

        total_tokens = handler.count_tokens(texts)
        logger.debug(f"Embedding complete: dimensions={result.dimensions}, tokens={total_tokens}")

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed_sparse", response_model=EmbeddingResponse)
async def create_sparse_embeddings(
    request: EmbeddingRequest,
    handler: BaseModelHandler = Depends(get_model_handler),
) -> EmbeddingResponse:
    try:
        texts = [request.input] if isinstance(request.input, str) else request.input
        logger.info(f"Creating sparse embeddings for {len(texts)} texts, model={request.model}")

        batch_size = handler.config["batch_size"]
        max_length = handler.config["max_length"]

        result = await handler.encode_multi(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            return_sparse=True,
        )

        data = []
        if result.sparse_embeddings:
            for i, sparse_obj in enumerate(result.sparse_embeddings):
                sparse_emb = SparseEmbedding(
                    indices=sparse_obj.indices,
                    values=sparse_obj.values,
                )
                embedding_data = EmbeddingData(
                    dense=[],
                    sparse=sparse_emb,
                )
                data.append(
                    EmbeddingObject(
                        object="embedding",
                        embedding=embedding_data,
                        index=i,
                    )
                )

        total_tokens = handler.count_tokens(texts)
        logger.debug(f"Sparse embedding complete: tokens={total_tokens}")

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except Exception as e:
        logger.error(f"Error creating sparse embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models() -> dict[str, Any]:
    logger.debug("Listing available models")
    handler = get_model_handler()
    model_id = getattr(handler, "model_id", handler.config.get("model_name", "unknown"))
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "embed-kit",
            }
        ],
    }
