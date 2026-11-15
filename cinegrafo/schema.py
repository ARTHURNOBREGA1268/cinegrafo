"""DDL idempotente; execução explícita via UI, nunca no entrypoint."""
import time
from .embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_PROPERTY, VECTOR_INDEX_NAME

SCHEMA = (
    "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX entity_canonical IF NOT EXISTS FOR (n:Entity) ON (n.canonical)",
    "CREATE INDEX chunk_created IF NOT EXISTS FOR (n:Chunk) ON (n.created_at)",
    "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
    f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS FOR (n:Chunk) ON (n.{EMBEDDING_PROPERTY}) "
    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBEDDING_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}",
)


def ensure_schema(session, timeout_seconds: int = 600):
    for statement in SCHEMA:
        session.run(statement).consume()

    deadline = time.monotonic() + timeout_seconds
    while True:
        indexes = [dict(row) for row in session.run(
            "SHOW INDEXES YIELD name, state, populationPercent RETURN name, state, populationPercent"
        )]
        failed = [row["name"] for row in indexes if row["state"] == "FAILED"]
        if failed:
            raise RuntimeError("Falha ao criar índices Neo4j: " + ", ".join(failed))

        pending = [row for row in indexes if row["state"] != "ONLINE"]
        if not pending:
            break
        if time.monotonic() >= deadline:
            details = ", ".join(
                f"{row['name']} ({row['state']}, {row['populationPercent']}%)" for row in pending
            )
            raise TimeoutError(f"Índices ainda não ficaram ONLINE após {timeout_seconds}s: {details}")
        time.sleep(1)
    return indexes
