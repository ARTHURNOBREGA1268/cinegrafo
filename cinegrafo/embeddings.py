from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding

from .config import settings

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_PROPERTY = "embedding_local"
VECTOR_INDEX_NAME = "chunk_embedding_local"


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Carrega o modelo ONNX local uma vez e conserva os pesos no volume persistente."""
    cache_dir = Path(settings.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(cache_dir),
        threads=settings.embedding_threads,
    )


class LocalEmbeddings:
    """Embeddings multilíngues executados localmente via FastEmbed/ONNX Runtime."""

    def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if input_type not in {"query", "document"}:
            raise ValueError("input_type deve ser 'query' ou 'document'.")
        if not texts:
            return []
        clean_texts = [str(text or "") for text in texts]
        vectors = [vector.tolist() for vector in _model().embed(clean_texts, batch_size=32)]
        if len(vectors) != len(clean_texts):
            raise RuntimeError("O modelo local retornou quantidade inesperada de vetores.")
        if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
            raise RuntimeError(f"O modelo local retornou embedding com dimensão diferente de {EMBEDDING_DIMENSIONS}.")
        if any(not all(isinstance(value, (int, float)) for value in vector) for vector in vectors):
            raise RuntimeError("O modelo local retornou embedding com valores não numéricos.")
        return [[float(value) for value in vector] for vector in vectors]
