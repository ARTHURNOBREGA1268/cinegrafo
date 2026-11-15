from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("GRAPH_DATABASE", "neo4j")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    anthropic_api_url: str = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages").rstrip("/")
    anthropic_version: str = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", "/app/data/embedding_models")
    embedding_threads: int = max(1, int(os.getenv("EMBEDDING_THREADS", "2")))
    chunk_fulltext_index: str = os.getenv("CHUNK_FULLTEXT_INDEX", "chunk_fulltext")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))


settings = Settings()
