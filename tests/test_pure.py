import json
import re
import pytest
from dataclasses import replace
from pathlib import Path
from cinegrafo.documents import split_chunks, extract_text
from cinegrafo.graph import (
    canonicalize, stable_entity_id, ENTITY_CREATE_TEMPLATES,
    RELATION_CREATE_TEMPLATES, DOMAIN_RELATION_PATTERN, GraphStore,
    build_fulltext_queries, select_shortest_path_endpoints,
)
from cinegrafo.llm import Claude
from cinegrafo.schema import SCHEMA
from cinegrafo.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_PROPERTY, VECTOR_INDEX_NAME, EMBEDDING_MODEL, LocalEmbeddings


def test_chunk_overlap_and_edges():
    assert split_chunks("um dois três quatro", size=3, overlap=1) == ["um dois três", "três quatro"]
    assert split_chunks("registro um\n\nregistro dois", size=3, overlap=1) == ["registro um", "registro dois"]
    assert split_chunks("") == []
    with pytest.raises(ValueError):
        split_chunks("x", size=2, overlap=2)


def test_demo_corpus_contains_100_unique_synthetic_works():
    corpus = Path(__file__).resolve().parents[1] / "docs" / "examples" / "corpus"
    files = sorted(corpus.glob("*.txt"))
    paragraphs = [p.strip() for file in files for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    titles = [re.search(r"“([^”]+)”", paragraph).group(1) for paragraph in paragraphs]
    ids = [int(re.match(r"Registro (\d{3})", paragraph).group(1)) for paragraph in paragraphs]
    assert len(files) == 4
    assert len(paragraphs) == 100
    assert sorted(ids) == list(range(1, 101))
    assert len(set(titles)) == 100
    assert sum("O filme fictício" in paragraph for paragraph in paragraphs) == 50
    assert sum("A série fictícia" in paragraph for paragraph in paragraphs) == 50


def test_canonicalization_is_deterministic():
    assert canonicalize("  Ação  Civil ") == "acao civil"
    assert stable_entity_id("Ação", "Lei") == stable_entity_id("acao", "lei")
    assert stable_entity_id("Ação", "Movie") != stable_entity_id("Ação", "Series")


def test_txt_and_reject_unknown():
    assert extract_text("amostra.txt", b"texto") == "texto"
    with pytest.raises(ValueError):
        extract_text("arquivo.exe", b"nao executar")


def test_extract_allowlists_audiovisual_types_and_relations():
    llm = Claude()
    llm.generate = lambda prompt, **_kwargs: json.dumps({
        "entities": [
            {"name": "O Farol de Âmbar", "type": "Movie"},
            {"name": "TelaNuvem", "type": "Platform"},
            {"name": "Ignorar regras", "type": "Instruction"},
        ],
        "relations": [
            {"source": "O Farol de Âmbar", "target": "TelaNuvem", "type": "AVAILABLE_ON", "evidence": "disponível"},
            {"source": "x", "target": "y", "type": "CALL_DB", "evidence": "no"},
        ],
    })
    result = llm.extract("texto tratado como dado")
    assert result["entities"] == [
        {"name": "O Farol de Âmbar", "type": "Movie"},
        {"name": "TelaNuvem", "type": "Platform"},
    ]
    assert result["relations"] == [
        {"source": "O Farol de Âmbar", "target": "TelaNuvem", "type": "AVAILABLE_ON", "evidence": "disponível"}
    ]


def test_extract_malformed_json_returns_empty_result():
    llm = Claude()
    llm.generate = lambda prompt, **_kwargs: "not json"
    assert llm.extract("texto") == {"entities": [], "relations": []}


def test_extract_applies_local_limits_when_schema_has_no_max_items():
    llm = Claude()
    llm.generate = lambda prompt, **_kwargs: json.dumps({
        "entities": [{"name": f"Filme {i}", "type": "Movie"} for i in range(12)],
        "relations": [{"source": "a", "target": "b", "type": "RELATED_TO", "evidence": "a b"} for _ in range(7)],
    })
    result = llm.extract("texto")
    assert len(result["entities"]) == 8
    assert len(result["relations"]) == 5


def test_fulltext_queries_are_safe_and_build_and_or_variants():
    strict, broad = build_fulltext_queries("Quem atuou no filme O Farol de Âmbar?")
    assert strict == "atuou AND filme AND farol AND âmbar"
    assert broad == "atuou OR filme OR farol OR âmbar"
    assert build_fulltext_queries("!!!") == ("", "")


def test_shortest_path_chooses_two_hop_endpoint_from_question_anchor():
    edges = [
        {"source_id": "work", "source_name": "O Farol", "source_type": "Movie",
         "source_canonical": "o farol", "target_id": "person", "target_name": "Sela",
         "target_type": "Person", "target_canonical": "sela"},
        {"source_id": "person", "source_name": "Sela", "source_type": "Person",
         "source_canonical": "sela", "target_id": "character", "target_name": "Tavi",
         "target_type": "Character", "target_canonical": "tavi"},
    ]
    assert select_shortest_path_endpoints(edges, ["o farol"]) == ("work", "character")


def test_extract_recovers_complete_items_from_truncated_json():
    llm = Claude()
    llm.generate = lambda prompt, **_kwargs: (
        '{"entities":['
        '{"name":"O Farol de Âmbar","type":"Movie"},'
        '{"name":"O Farol de Âmbar","type":"Series"},'
        '{"name":"Sela Orin","type":"Person"}],'
        '"relations":[{"source":"Sela Orin","target":"O Farol de Âmbar",'
        '"type":"ACTED_IN","evidence":"Sela Orin atuou em O Farol de Âmbar"},'
        '{"source":"incompleto"'
    )

    result = llm.extract("texto")
    assert result["entities"] == [
        {"name": "O Farol de Âmbar", "type": "Movie"},
        {"name": "Sela Orin", "type": "Person"},
    ]
    assert result["relations"] == [{
        "source": "Sela Orin", "target": "O Farol de Âmbar",
        "type": "ACTED_IN", "evidence": "Sela Orin atuou em O Farol de Âmbar",
    }]


def test_extract_uses_claude_structured_json_schema(monkeypatch):
    captured = {}
    from cinegrafo.config import settings
    monkeypatch.setattr("cinegrafo.llm.settings", replace(
        settings, anthropic_api_key="test-secret", anthropic_model="claude-test-model"
    ))

    class Response:
        ok = True
        def json(self):
            return {"content": [{"type": "text", "text": '{"entities": [], "relations": []}'}]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("cinegrafo.llm.requests.post", fake_post)
    result = Claude().extract("texto curto")

    assert result == {"entities": [], "relations": []}
    assert captured["url"].endswith("/v1/messages")
    schema = captured["payload"]["output_config"]["format"]["schema"]
    assert schema["required"] == ["entities", "relations"]
    assert "maxItems" not in schema["properties"]["entities"]
    assert "maxItems" not in schema["properties"]["relations"]
    assert captured["payload"]["max_tokens"] == 512
    assert captured["headers"]["Authorization"] == "Bearer test-secret"


def test_claude_messages_api_uses_runtime_key_and_schema(monkeypatch):
    from cinegrafo.llm import Claude

    from cinegrafo.config import settings
    monkeypatch.setattr("cinegrafo.llm.settings", replace(
        settings, anthropic_api_key="test-secret",
        anthropic_model="claude-test-model",
    ))
    captured = {}

    class Response:
        ok = True
        def json(self):
            return {"content": [{"type": "text", "text": "Resposta Claude"}]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("cinegrafo.llm.requests.post", fake_post)
    answer = Claude().generate("pergunta", max_tokens=123, json_schema={"type": "object"})

    assert answer == "Resposta Claude"
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["payload"]["model"] == "claude-test-model"
    assert captured["payload"]["max_tokens"] == 123
    assert captured["payload"]["output_config"]["format"]["schema"] == {"type": "object"}


def test_claude_empty_text_error_reports_safe_response_metadata(monkeypatch):
    from cinegrafo.config import settings
    monkeypatch.setattr("cinegrafo.llm.settings", replace(
        settings, anthropic_api_key="test-secret",
    ))

    class Response:
        ok = True
        def json(self):
            return {
                "id": "msg_test_123", "stop_reason": "max_tokens",
                "content": [{"type": "thinking", "thinking": "private internal text"}],
            }

    monkeypatch.setattr("cinegrafo.llm.requests.post", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError) as err:
        Claude().generate("test")
    message = str(err.value)
    assert "stop_reason=max_tokens" in message
    assert "thinking" in message
    assert "msg_test_123" in message
    assert "private internal text" not in message


def test_claude_has_no_embedding_or_local_provider_fallback():
    from cinegrafo.config import settings
    assert not hasattr(Claude(), "embed")
    assert not hasattr(settings, "ollama_url")
    assert not hasattr(settings, "llm_provider")


def test_local_embeddings_encode_without_network_and_validate_dimensions(monkeypatch):
    import numpy as np
    from cinegrafo.embeddings import _model

    class FakeModel:
        def embed(self, texts, batch_size):
            assert texts == ["filme em português", "pergunta"]
            assert batch_size == 32
            return iter([np.ones(EMBEDDING_DIMENSIONS), np.zeros(EMBEDDING_DIMENSIONS)])

    monkeypatch.setattr("cinegrafo.embeddings._model", lambda: FakeModel())
    vectors = LocalEmbeddings().embed(["filme em português", "pergunta"], input_type="document")
    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS == 384
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 0.0
    assert EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_local_embeddings_empty_input_and_mode_validation():
    embeddings = LocalEmbeddings()
    assert embeddings.embed([], input_type="query") == []
    with pytest.raises(ValueError, match="input_type"):
        embeddings.embed(["texto"], input_type="wrong")


def test_lpg_templates_only_use_allowlisted_real_types():
    from cinegrafo.llm import ENTITY_TYPES, RELATION_TYPES

    assert set(ENTITY_CREATE_TEMPLATES) == ENTITY_TYPES
    assert set(RELATION_CREATE_TEMPLATES) == RELATION_TYPES
    for kind, query in ENTITY_CREATE_TEMPLATES.items():
        assert f":Entity:{kind}" in query
        assert "e.type=$type" in query
    for kind, query in RELATION_CREATE_TEMPLATES.items():
        assert f"[r:{kind}]" in query
        assert "kind:" not in query
        assert "evidence_chunk_ids" in query
        assert "evidence_chunk_id=$cid" in query
    assert set(DOMAIN_RELATION_PATTERN.split("|")) == RELATION_TYPES
    assert "RELATED" not in set(DOMAIN_RELATION_PATTERN.split("|"))
    assert "HAS_CHUNK" not in DOMAIN_RELATION_PATTERN
    assert "MENTIONS" not in DOMAIN_RELATION_PATTERN


def test_untrusted_type_cannot_be_used_as_cypher_identifier():
    assert ENTITY_CREATE_TEMPLATES.get("Movie}) MATCH (n) DETACH DELETE n //") is None
    assert RELATION_CREATE_TEMPLATES.get("ACTED_IN|RELATED") is None


def test_retrieve_uses_hybrid_search_and_static_cypher_expansion():
    edge = {
        "source_id": "person-1", "source_name": "Sela Orin", "source_type": "Person",
        "source_canonical": "sela orin",
        "target_id": "movie-1", "target_name": "O Farol de Âmbar", "target_type": "Movie",
        "target_canonical": "o farol de ambar",
        "kind": "ACTED_IN", "evidence": "Sela Orin atuou em O Farol de Âmbar",
        "confidence": 0.5, "evidence_source_id": "source-1", "evidence_chunk_id": "chunk-1",
        "evidence_chunk_ids": ["chunk-1"], "evidence_quotes": [],
        "evidence_chunk_text": "Sela Orin atuou em O Farol de Âmbar",
        "evidence_source_title": "demo.txt", "retrieval_chunk_id": "chunk-1",
    }
    chunk = {"id": "chunk-1", "text": "Sela Orin atuou em O Farol de Âmbar",
             "source_id": "source-1", "title": "demo.txt", "score": 0.9}

    class Tx:
        def run(self, query, **_params):
            assert "UNWIND $ids AS cid" in query
            return [edge]

    class Session:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def run(self, query, **_params):
            if "db.index.vector.queryNodes" in query:
                return [{**chunk, "score": 0.95}]
            if "db.index.fulltext.queryNodes" in query:
                return [chunk]
            if "RETURN DISTINCT e.canonical" in query:
                return [{"canonical": "o farol de ambar"}]
            if "shortestPath" in query:
                return [{"nodes": [{"id": "person-1", "name": "Sela Orin", "type": "Person"},
                                   {"id": "movie-1", "name": "O Farol de Âmbar", "type": "Movie"}],
                         "rels": [{"kind": "ACTED_IN", "evidence": edge["evidence"],
                                   "confidence": 0.5, "source_id": "source-1",
                                   "evidence_chunk_id": "chunk-1"}]}]
            return []
        def execute_read(self, callback):
            return callback(Tx())

    class FakeClaude:
        pass

    class FakeEmbeddings:
        def embed(self, texts, *, input_type):
            assert input_type == "query"
            assert texts == ["Quem atuou no filme O Farol de Âmbar?"]
            return [[0.1] * EMBEDDING_DIMENSIONS]

    db = GraphStore.__new__(GraphStore)
    db.llm = FakeClaude()
    db.embeddings = FakeEmbeddings()
    db.session = lambda: Session()
    result = db.retrieve("Quem atuou no filme O Farol de Âmbar?", 1)
    assert result["graph"] == [edge]
    assert "FastEmbed local" in result["retrieval_note"]
    assert "similaridade vetorial" in result["retrieval_note"]
    assert "expansão Cypher domain" in result["cypher"]
    assert result["chunks"][0]["retrieval_sources"] == ["vector", "fulltext_and", "fulltext_or"]
    assert "db.index.vector.queryNodes" in result["cypher"]
    assert "size($anchors)=0 OR seed.canonical IN $anchors" in result["cypher"]
    assert result["shortest_path"][0]["rels"][0]["kind"] == "ACTED_IN"


def test_schema_has_fulltext_range_and_local_vector_indexes_without_dropping_legacy_data():
    statements = "\n".join(SCHEMA)
    assert "CREATE INDEX entity_canonical" in statements
    assert "CREATE FULLTEXT INDEX chunk_fulltext" in statements
    assert "ON EACH [n.text]" in statements
    assert f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME}" in statements
    assert f"n.{EMBEDDING_PROPERTY}" in statements
    assert str(EMBEDDING_DIMENSIONS) in statements
    assert "vector.similarity_function`: 'cosine'" in statements
    assert "DROP INDEX chunk_embedding IF EXISTS" not in statements
    assert "REMOVE c.embedding" not in statements


def test_ingestion_is_idempotent_and_rejects_nonliteral_relation_evidence():
    class Result:
        def consume(self):
            return None

    class Session:
        def __init__(self):
            self.calls = []
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def run(self, query, **parameters):
            self.calls.append((query, parameters))
            return Result()

    class FakeLLM:
        def extract(self, _body):
            return {
                "entities": [
                    {"name": "Sela Orin", "type": "Person"},
                    {"name": "O Farol de Âmbar", "type": "Movie"},
                ],
                "relations": [
                    {"source": "Sela Orin", "target": "O Farol de Âmbar", "type": "ACTED_IN",
                     "evidence": "Sela Orin atuou no filme O Farol de Âmbar"},
                    {"source": "Sela Orin", "target": "O Farol de Âmbar", "type": "DIRECTED",
                     "evidence": "texto que não consta no documento"},
                ],
            }

    session = Session()
    db = GraphStore.__new__(GraphStore)
    db.llm = FakeLLM()
    class FakeEmbeddings:
        def embed(self, texts, *, input_type):
            assert input_type == "document"
            return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
    db.embeddings = FakeEmbeddings()
    db.session = lambda: session
    text = "Sela Orin atuou no filme O Farol de Âmbar."
    source_a, chunks_a = db.ingest("demo.txt", text)
    source_b, chunks_b = db.ingest("demo.txt", text)

    assert source_a == source_b
    assert chunks_a == chunks_b == 1
    relation_calls = [(q, p) for q, p in session.calls if "MERGE (a)-[r:" in q]
    assert len(relation_calls) == 2  # Uma relação ACTED_IN por reingestão; DIRECTED sem citação foi ignorada.
    query, params = relation_calls[0]
    assert "MERGE (a)-[r:ACTED_IN]" in query
    assert params["evidence"] in text
    assert params["cid"].startswith(source_a + ":")
    chunk_calls = [(q, p) for q, p in session.calls if "MERGE (c:Chunk" in q]
    assert len(chunk_calls) == 2
    assert f"c.{EMBEDDING_PROPERTY}=$embedding" in chunk_calls[0][0]
    assert len(chunk_calls[0][1]["embedding"]) == EMBEDDING_DIMENSIONS
    assert "a_id" in params and "b_id" in params


def test_explicit_relation_rules_recover_creator_and_multi_hop_facts_safely():
    from cinegrafo.relation_rules import infer_explicit_relations

    text = (
        "A série fictícia “Jardins de Mercúrio” estreou em 2006, tem gênero comédia "
        "e integra a franquia fictícia “Crônicas de Nébula”. "
        "Ivo Maren criou a série “Jardins de Mercúrio”. "
        "Sela Orin atuou na série “Jardins de Mercúrio” como a personagem fictícia “Ena Varo”."
    )
    entities = [
        {"name": "Jardins de Mercúrio", "type": "Series"},
        {"name": "Ivo Maren", "type": "Person"},
        {"name": "Sela Orin", "type": "Person"},
        {"name": "Ena Varo", "type": "Character"},
        {"name": "comédia", "type": "Genre"},
        {"name": "Crônicas de Nébula", "type": "Franchise"},
    ]
    relations = infer_explicit_relations(text, entities)
    triples = {(r["source"], r["target"], r["type"]) for r in relations}
    assert ("Ivo Maren", "Jardins de Mercúrio", "CREATED") in triples
    assert ("Sela Orin", "Jardins de Mercúrio", "ACTED_IN") in triples
    assert ("Sela Orin", "Ena Varo", "PORTRAYS") in triples
    assert ("Jardins de Mercúrio", "comédia", "HAS_GENRE") in triples
    assert ("Jardins de Mercúrio", "Crônicas de Nébula", "PART_OF") in triples
    assert all(r["evidence"] in text for r in relations)


def test_explicit_relation_rules_do_not_link_entities_from_neighboring_sentences():
    from cinegrafo.relation_rules import infer_explicit_relations

    text = "Ivo Maren criou uma obra sem nome. Jardins de Mercúrio é uma série de fantasia."
    entities = [
        {"name": "Ivo Maren", "type": "Person"},
        {"name": "Jardins de Mercúrio", "type": "Series"},
    ]
    assert infer_explicit_relations(text, entities) == []


def test_backfill_explicit_relations_repairs_stored_chunks_without_llm():
    class Result:
        def consume(self):
            return None

    text = (
        "A série fictícia “Jardins de Mercúrio” estreou em 2006, tem gênero comédia "
        "e integra a franquia fictícia “Crônicas de Nébula”. "
        "Ivo Maren criou a série “Jardins de Mercúrio”. "
        "Sela Orin atuou na série “Jardins de Mercúrio” como a personagem fictícia “Ena Varo”."
    )
    entities = [
        {"name": "Jardins de Mercúrio", "type": "Series"},
        {"name": "Ivo Maren", "type": "Person"},
        {"name": "Sela Orin", "type": "Person"},
        {"name": "Ena Varo", "type": "Character"},
        {"name": "comédia", "type": "Genre"},
        {"name": "Crônicas de Nébula", "type": "Franchise"},
    ]

    class Session:
        def __init__(self):
            self.relations = []
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def run(self, query, **params):
            if "MATCH (c:Chunk)" in query:
                if params["skip"]:
                    return []
                return [{"id": "chunk-1", "text": text, "source_id": "source-1", "entities": entities}]
            if "MERGE (a)-[r:" in query:
                self.relations.append((query, params))
            return Result()

    session = Session()
    db = GraphStore.__new__(GraphStore)
    db.session = lambda: session
    assert db.backfill_explicit_relations(batch_size=1) == 5
    assert len(session.relations) == 5
    created = next((query, params) for query, params in session.relations if "[r:CREATED]" in query)
    assert created[1]["sid"] == "source-1"
    assert created[1]["cid"] == "chunk-1"
    assert created[1]["model"] == "deterministic-rule-v1"
