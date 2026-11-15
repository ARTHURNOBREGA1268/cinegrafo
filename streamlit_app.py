from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from cinegrafo.config import settings
from cinegrafo.documents import extract_text
from cinegrafo.graph import GraphStore
from cinegrafo.schema import ensure_schema

st.set_page_config(page_title="CineGrafo", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1380px; padding-top: 1.25rem; padding-bottom: 2rem;}
    h1 {letter-spacing: -.035em; font-weight: 720; margin-bottom: .1rem;}
    h2, h3 {letter-spacing: -.02em;}
    [data-testid="stCaptionContainer"] {line-height: 1.45;}
    [data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stDataFrame"] {
        border-radius: 10px;
    }
    [data-testid="stButton"] button {border-radius: 8px; min-height: 2.55rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 140, 160, .18);}
    div[data-testid="stMetric"] {padding: .75rem .85rem; border: 1px solid rgba(128, 140, 160, .18); border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Quem atua nas duas obras da franquia Crônicas de Nébula, quais personagens interpreta e quais gêneros são associados a cada obra?",
    "Quem atuou no filme O Farol de Âmbar, qual personagem interpretou e em qual franquia o filme está?",
    "Quem criou Jardins de Mercúrio, quem atua na série e qual personagem essa pessoa interpreta?",
    "Quais as duas obras do catálogo na plataforma TelaNuvem, que gêneros são associados a cada uma e qual pessoa criadora elas compartilham?",
    "Quais temporadas estão ligadas à série e a qual franquia as obras pertencem?",
]
MAX_DISPLAY_EDGES = 24


@st.cache_resource
def store() -> GraphStore:
    return GraphStore()


def show_ingestion_summary(completed: list[tuple], errors: list[tuple]) -> None:
    if completed:
        total_chunks = sum(item[2] for item in completed)
        total_entities = sum(item[3] for item in completed)
        total_relations = sum(item[4] for item in completed)
        st.success(
            f"Concluído: {len(completed)} documentos · {total_chunks} trechos · "
            f"{total_entities} entidades · {total_relations} relações."
        )
        with st.expander("Ver resumo por documento", expanded=False):
            st.dataframe(
                [
                    {"Documento": name, "Trechos": chunks, "Entidades": entities, "Relações": relations}
                    for name, _source_id, chunks, entities, relations in completed
                ],
                width="stretch",
                hide_index=True,
            )
        empty = [name for name, _sid, _chunks, entities, relations in completed if not entities or not relations]
        if empty:
            st.warning("Sem entidades ou relações validadas em: " + ", ".join(empty))
    if errors:
        st.error("Falhas: " + "; ".join(f"{name} ({detail})" for name, detail in errors))


def ingest_payloads(db: GraphStore, payloads: list[tuple[str, bytes]], label: str) -> None:
    progress = st.progress(0.0, text=f"Preparando {label.lower()}…")
    completed, errors = [], []
    for index, (name, payload) in enumerate(payloads, 1):
        try:
            text = extract_text(os.path.basename(name), payload)
            if not text.strip():
                raise ValueError("O documento não contém texto extraível.")
            source_id, chunks = db.ingest(os.path.basename(name), text)
            stats = db.source_summary(source_id)
            completed.append((os.path.basename(name), source_id, chunks, stats["entities"], stats["relations"]))
        except Exception as exc:
            errors.append((os.path.basename(name), f"{type(exc).__name__}: {exc}"))
        progress.progress(index / len(payloads), text=f"Processados {index} de {len(payloads)} documentos…")
    progress.empty()
    show_ingestion_summary(completed, errors)


def render_ingestion(db: GraphStore) -> None:
    st.header("Ingestão")
    st.caption("Adicione fontes de filmes e séries ao grafo. Formatos aceitos: TXT, PDF e DOCX.")

    documents = st.file_uploader(
        "Selecione documentos",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="graph_documents",
        help=f"Até {settings.max_upload_mb} MB por arquivo. Uma fonte reimportada não é duplicada.",
    )
    oversized = [file.name for file in (documents or []) if file.size > settings.max_upload_mb * 1024 * 1024]
    if oversized:
        st.error(f"Limite de {settings.max_upload_mb} MB excedido: " + ", ".join(oversized))

    if st.button(
        "Ingerir documentos selecionados",
        type="primary",
        disabled=not documents or bool(oversized) or not settings.anthropic_api_key,
        key="ingest_uploaded",
    ):
        ingest_payloads(db, [(file.name, file.getvalue()) for file in documents], "documentos")

    demo_files = sorted(Path("docs/examples/corpus").glob("*.txt"))
    if demo_files:
        demo_records = sum(path.read_text(encoding="utf-8").count("Registro ") for path in demo_files)
        with st.expander("Corpus de demonstração", expanded=False):
            st.caption(f"{len(demo_files)} arquivos TXT · {demo_records} obras fictícias.")
            st.warning(
                "A ingestão chama Claude para extrair relações e pode consumir cota ou gerar cobrança. "
                "Embeddings são calculados localmente."
            )
            if st.button(
                "Ingerir corpus sintético",
                disabled=not settings.anthropic_api_key,
                key="ingest_demo",
            ):
                ingest_payloads(db, [(path.name, path.read_bytes()) for path in demo_files], "corpus")
    else:
        st.caption("Corpus de demonstração não encontrado na imagem do aplicativo.")

    with st.expander("Manutenção da base", expanded=False):
        st.caption("Ações para dados existentes. Não apagam fontes, trechos ou relações.")
        col_embeddings, col_relations = st.columns(2)
        with col_embeddings:
            st.markdown("**Vetores locais**")
            st.caption("Completa embeddings ausentes nos trechos já armazenados.")
            if st.button("Gerar embeddings faltantes", key="backfill_local"):
                progress = st.progress(0.0, text="Preparando vetorização local…")
                try:
                    with st.spinner("Calculando embeddings com FastEmbed…"):
                        def update_progress(done: int, total: int) -> None:
                            progress.progress(
                                min(done / total, 1.0) if total else 1.0,
                                text=f"Vetorizados {done} de {total} trechos…",
                            )
                        count = db.backfill_embeddings(progress_callback=update_progress)
                    progress.empty()
                    st.success(f"Vetores locais verificados: {count} trechos atualizados." if count else "Todos os trechos já possuem vetor local.")
                except Exception as exc:
                    st.error(f"Falha na vetorização: {type(exc).__name__}: {exc}")
        with col_relations:
            st.markdown("**Relações explícitas**")
            st.caption("Recupera relações textualmente confirmadas, sem chamar Claude.")
            if st.button("Verificar relações existentes", key="repair_relations"):
                try:
                    with st.spinner("Verificando evidências nos trechos já armazenados…"):
                        repaired = db.backfill_explicit_relations()
                    st.success(f"{repaired} relações verificadas ou atualizadas. Nenhuma chamada Claude foi feita.")
                except Exception as exc:
                    st.error(f"Falha ao verificar relações: {type(exc).__name__}: {exc}")


def render_graph(db: GraphStore, question: str, k: int) -> None:
    with st.spinner("Recuperando evidências e construindo a resposta…"):
        result = db.retrieve(question, k)

    if result["chunks"]:
        try:
            with st.spinner("Redigindo resposta a partir das evidências…"):
                answer = db.answer(question, result)
        except Exception as exc:
            answer = None
            st.warning(f"A geração da resposta falhou ({type(exc).__name__}). As evidências continuam disponíveis abaixo.")
        with st.container(border=True):
            st.markdown("#### Resposta")
            if answer:
                st.write(answer)
            else:
                st.info("Consulte os trechos recuperados e as relações abaixo.")
    else:
        st.info("Nenhuma evidência foi recuperada. Verifique se o documento que contém a resposta foi ingerido.")

    graph_rows = result["graph"]
    node_map: dict[str, dict] = {}
    for row in graph_rows:
        for prefix in ("source", "target"):
            node_id = row.get(f"{prefix}_id")
            if node_id:
                node_map[node_id] = {
                    "id": node_id,
                    "name": row.get(f"{prefix}_name") or "",
                    "type": row.get(f"{prefix}_type") or "",
                }

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Trechos de evidência", len(result["chunks"]))
    metric_b.metric("Entidades no subgrafo", len(node_map))
    metric_c.metric("Relações recuperadas", len(graph_rows))

    if graph_rows:
        st.markdown("### Subgrafo da consulta")
        plot_rows = graph_rows[:MAX_DISPLAY_EDGES]
        graph = nx.Graph()
        for row in plot_rows:
            graph.add_edge(row["source_id"], row["target_id"])
        positions = nx.spring_layout(graph, seed=42, k=1.35, iterations=90)
        edge_x, edge_y, edge_hover = [], [], []
        for row in plot_rows:
            source, target = row["source_id"], row["target_id"]
            if source not in positions or target not in positions:
                continue
            edge_x.extend([positions[source][0], positions[target][0], None])
            edge_y.extend([positions[source][1], positions[target][1], None])
            label = f"{row.get('source_name')} — {row.get('kind')} → {row.get('target_name')}"
            evidence = row.get("evidence") or "Sem citação textual anexada"
            edge_hover.extend([f"{label}<br>{evidence}"] * 3)
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.4, "color": "#65738b"},
            hovertext=edge_hover,
            hoverinfo="text",
            name="Relações",
        )
        nodes = list(graph.nodes())
        node_labels = [f"{node_map[n]['name']} · {node_map[n]['type']}" for n in nodes]
        node_trace = go.Scatter(
            x=[positions[n][0] for n in nodes],
            y=[positions[n][1] for n in nodes],
            mode="markers+text",
            text=node_labels if len(nodes) <= 14 else [""] * len(nodes),
            textposition="top center",
            hovertext=node_labels,
            hoverinfo="text",
            marker={"size": 20, "color": "#60a5fa", "line": {"width": 1, "color": "#dbeafe"}},
            name="Entidades",
        )
        figure = go.Figure([edge_trace, node_trace])
        figure.update_layout(
            showlegend=False,
            height=450,
            margin={"l": 8, "r": 8, "t": 8, "b": 8},
            hovermode="closest",
            dragmode="pan",
            xaxis={"visible": False},
            yaxis={"visible": False},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            uirevision="cinegrafo-graph",
        )
        st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displayModeBar": True})
        if len(graph_rows) > len(plot_rows):
            st.caption(f"Visualização resumida: {len(plot_rows)} das {len(graph_rows)} relações recuperadas. Passe o cursor sobre nós e arestas para ver detalhes.")
        else:
            st.caption("Passe o cursor sobre nós e arestas para consultar nomes, tipos e evidências.")
    else:
        st.info("Nenhuma relação de domínio alcançada para esta consulta.")

    with st.expander("Evidências textuais", expanded=False):
        if result["chunks"]:
            for index, item in enumerate(result["chunks"], 1):
                title = f"{index}. {item['title']} · RRF {item['rrf_score']:.4f}"
                st.markdown(f"**{title}**")
                st.caption(f"Chunk `{item['id']}` · recuperado por {', '.join(item['retrieval_sources'])}")
                st.write(item["text"])
        else:
            st.caption("Nenhum trecho recuperado.")

    with st.expander("Relações, nós e proveniência", expanded=False):
        if graph_rows:
            relation_table = [
                {
                    "Origem": row.get("source_name"),
                    "Tipo": row.get("kind"),
                    "Destino": row.get("target_name"),
                    "Confiança*": row.get("confidence"),
                    "Evidência": row.get("evidence"),
                    "Fonte": row.get("evidence_source_title"),
                }
                for row in graph_rows
            ]
            st.dataframe(relation_table, width="stretch", hide_index=True)
            st.caption("* Confiança 0,5 é heurística fixa de demonstração, não probabilidade calibrada.")
        else:
            st.caption("Nenhuma relação disponível.")
        if node_map:
            st.markdown("**Nós consultados**")
            st.dataframe(
                [{"Nome": item["name"], "Tipo": item["type"], "ID": item["id"]} for item in node_map.values()],
                width="stretch",
                hide_index=True,
            )

    with st.expander("Caminho mínimo e detalhes da recuperação", expanded=False):
        if result["shortest_path"]:
            st.markdown("**Caminho mínimo (até dois saltos)**")
            st.json(result["shortest_path"])
        else:
            st.caption("Nenhum caminho de domínio foi encontrado para os endpoints desta consulta.")
        st.markdown("**Consulta Cypher executada**")
        st.code(result["cypher"], language="cypher")
        st.caption(result["retrieval_note"])


def render_query(db: GraphStore) -> None:
    st.header("Consulta GraphRAG")
    st.caption("Pergunte sobre filmes e séries; acompanhe a resposta até os trechos e relações que a sustentam.")

    with st.expander("Parâmetros de recuperação", expanded=False):
        k = st.slider("Trechos de evidência", min_value=1, max_value=10, value=5)

    with st.form("cinegrafo_query_form"):
        selected = st.selectbox(
            "Pergunta sugerida",
            ["Escrever uma pergunta…", *EXAMPLES],
            index=0,
        )
        typed_question = st.text_input(
            "Sua pergunta",
            placeholder="Ex.: Quem criou esta série e a qual franquia ela pertence?",
        )
        question = typed_question.strip() or ("" if selected.startswith("Escrever") else selected)
        submitted = st.form_submit_button(
            "Consultar evidências",
            type="primary",
            disabled=not settings.anthropic_api_key,
        )
    if not settings.anthropic_api_key:
        st.warning("Configure `ANTHROPIC_API_KEY` no `.env` para habilitar consultas e ingestão.")
    if submitted and question:
        try:
            render_graph(db, question, k)
        except Exception as exc:
            st.error(f"A consulta falhou: {type(exc).__name__}: {exc}")


def render_catalog(db: GraphStore) -> None:
    st.header("Catálogo de entidades")
    st.caption("Inventário dos dados extraídos nesta instalação do Neo4j.")
    with db.session() as session:
        totals = session.run("""
            CALL { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL { MATCH (s:Source) RETURN count(s) AS sources }
            CALL { MATCH (c:Chunk) RETURN count(c) AS chunks }
            RETURN entities,sources,chunks
        """).single()
        type_rows = session.run("""
            MATCH (e:Entity) RETURN e.type AS tipo,count(*) AS total ORDER BY total DESC
        """).data()
        source_rows = session.run("""
            MATCH (s:Source)-[:HAS_CHUNK]->(c:Chunk)
            RETURN s.title AS fonte,count(c) AS trechos ORDER BY fonte
        """).data()

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Entidades", f"{int(totals['entities']):,}")
    metric_b.metric("Fontes", f"{int(totals['sources']):,}")
    metric_c.metric("Trechos", f"{int(totals['chunks']):,}")

    if not type_rows:
        st.info("O catálogo está vazio. Ingira documentos para começar.")
        return

    with st.expander("Distribuição por tipo e fonte", expanded=False):
        left, right = st.columns(2)
        left.dataframe(
            [{"Tipo": row["tipo"], "Entidades": int(row["total"])} for row in type_rows],
            width="stretch",
            hide_index=True,
        )
        right.dataframe(
            [{"Fonte": row["fonte"], "Trechos": int(row["trechos"])} for row in source_rows],
            width="stretch",
            hide_index=True,
        )

    type_filter = st.selectbox(
        "Filtrar por tipo",
        ["Todos", *[row["tipo"] for row in type_rows]],
    )
    page_size = st.selectbox("Linhas por página", [25, 50, 100], index=1)
    with db.session() as session:
        total = int(session.run(
            "MATCH (e:Entity) WHERE $tipo='Todos' OR e.type=$tipo RETURN count(e) AS total",
            tipo=type_filter,
        ).single()["total"])
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
        rows = session.run("""
            MATCH (e:Entity) WHERE $tipo='Todos' OR e.type=$tipo
            RETURN e.name AS nome,e.type AS tipo,e.canonical AS canonico,e.id AS id
            ORDER BY e.name SKIP $skip LIMIT $limit
        """, tipo=type_filter, skip=(int(page) - 1) * page_size, limit=page_size).data()
    st.caption(f"{total:,} entidades · página {int(page)} de {total_pages}")
    st.dataframe(
        [{"Nome": row["nome"], "Tipo": row["tipo"], "Canônico": row["canonico"], "ID": row["id"]} for row in rows],
        width="stretch",
        hide_index=True,
        height=440,
    )


try:
    db = store()
    with st.sidebar:
        st.markdown("## CineGrafo")
        st.caption("GraphRAG · filmes e séries")
        page = st.radio(
            "Navegação",
            ["Consulta", "Ingestão", "Catálogo"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Serviços**")
        st.caption("Claude API · " + ("configurada" if settings.anthropic_api_key else "chave ausente"))
        st.caption("FastEmbed · vetores locais de 384 dimensões")
        with st.expander("Neo4j", expanded=False):
            st.caption("Neo4j Desktop fornece o banco; o Compose executa apenas o app.")
            if st.button("Preparar índices e constraints", key="ensure_schema"):
                try:
                    with st.spinner("Preparando schema e aguardando índices…"):
                        with db.session() as session:
                            indexes = ensure_schema(session)
                    st.success(f"Banco preparado · {len(indexes)} índices ONLINE.")
                except Exception as exc:
                    st.error(f"Falha ao preparar Neo4j: {type(exc).__name__}: {exc}")

    st.title("CineGrafo")
    st.caption("Explore relações entre obras, pessoas, personagens e gêneros — com evidências rastreáveis.")

    if page == "Consulta":
        render_query(db)
    elif page == "Ingestão":
        render_ingestion(db)
    else:
        render_catalog(db)
except Exception as exc:
    st.error(f"Aplicação indisponível: {type(exc).__name__}: {exc}. Confira `.env`, Neo4j e o acesso à Claude API.")
