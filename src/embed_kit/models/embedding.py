from typing import Any

from pydantic import BaseModel

from embed_kit.models.base import BaseModelHandler
from embed_kit.models.registry import registry
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.models.embedding")


class EmbeddingInput(BaseModel):
    texts: list[str]
    model: str | None = None
    batch_size: int = 12
    max_length: int = 8192


class EmbeddingOutput(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    num_texts: int


class EmbeddingAllOutput(BaseModel):
    dense_embeddings: list[list[float]]
    sparse_embeddings: list[dict[int, float]] | None
    model: str
    dimensions: int
    num_texts: int


class EmbeddingModelHandler(BaseModelHandler[EmbeddingInput, EmbeddingOutput]):
    @property
    def input_schema(self) -> type[EmbeddingInput]:
        return EmbeddingInput

    @property
    def output_schema(self) -> type[EmbeddingOutput]:
        return EmbeddingOutput

    async def process(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        model_name = input_data.model or self.config["model_name"]
        logger.debug(f"Processing {len(input_data.texts)} texts with model {model_name}")

        embeddings = await self._generate_embeddings(
            input_data.texts,
            model_name,
            input_data.batch_size,
            input_data.max_length,
        )
        dimensions = len(embeddings[0]) if embeddings else 0

        logger.debug(f"Generated embeddings: dimensions={dimensions}, count={len(embeddings)}")
        return EmbeddingOutput(
            embeddings=embeddings,
            model=model_name,
            dimensions=dimensions,
            num_texts=len(embeddings),
        )

    async def _generate_embeddings(
        self,
        texts: list[str],
        model: str,
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        raise NotImplementedError("Subclasses must implement _generate_embeddings")

    async def generate_embeddings_all(
        self,
        texts: list[str],
        model: str,
        batch_size: int,
        max_length: int,
        return_sparse: bool,
    ) -> EmbeddingAllOutput:
        raise NotImplementedError("Subclasses must implement generate_embeddings_all")


@registry.register_handler("bge-m3")
class BGEM3Handler(EmbeddingModelHandler):
    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._model = None
        logger.debug(f"BGEM3Handler initialized with config: {config}")

    async def _setup(self) -> None:
        from FlagEmbedding import BGEM3FlagModel

        model_path = self.config["model_path"]
        use_fp16 = self.config["use_fp16"]
        device = self.config["device"]
        normalize_embeddings = self.config["normalize_embeddings"]

        logger.info(f"Loading BGE-M3 model from {model_path}")
        logger.debug(f"Model config: use_fp16={use_fp16}, device={device}, normalize={normalize_embeddings}")

        self._model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )
        logger.info("BGE-M3 model loaded successfully")

    async def _generate_embeddings(
        self,
        texts: list[str],
        model: str,
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        if self._model is None:
            await self._setup()

        logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}, max_length={max_length}")

        result = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return result["dense_vecs"].tolist()

    async def generate_embeddings_all(
        self,
        texts: list[str],
        model: str,
        batch_size: int,
        max_length: int,
        return_sparse: bool,
    ) -> EmbeddingAllOutput:
        if self._model is None:
            await self._setup()

        logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}, max_length={max_length}, return_sparse={return_sparse}")

        result = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        dense_embeddings = result["dense_vecs"].tolist()
        sparse_embeddings = None

        if return_sparse:
            sparse_embeddings = []
            for i in range(len(texts)):
                sparse_dict = result["lexical_weights"][i]
                sparse_embeddings.append({int(k): float(v) for k, v in sparse_dict.items()})

        dimensions = len(dense_embeddings[0]) if dense_embeddings else 0

        return EmbeddingAllOutput(
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            model=model,
            dimensions=dimensions,
            num_texts=len(dense_embeddings),
        )

    async def health_check(self) -> dict[str, Any]:
        base_health = await super().health_check()
        base_health["model_path"] = self.config["model_path"]
        base_health["device"] = self.config["device"]
        return base_health

    def count_tokens(self, texts: list[str]) -> int:
        if self._model is None:
            logger.warning("Model not initialized, using word count as fallback")
            return sum(len(text.split()) for text in texts)

        total_tokens = 0
        for text in texts:
            tokens = self._model.tokenizer.encode(text)
            total_tokens += len(tokens)

        return total_tokens
