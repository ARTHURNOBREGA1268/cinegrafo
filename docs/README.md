# CineGrafo — documentação técnica

## Visão geral

CineGrafo é um protótipo acadêmico em Python 3.12, Streamlit, Neo4j, Anthropic Claude Messages API e FastEmbed local. Claude extrai entidades/relações de TXT/PDF/DOCX e redige respostas. FastEmbed/ONNX Runtime calcula vetores localmente usando `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões). Neo4j Desktop armazena o LPG e executa buscas vetoriais, full-text e Cypher. Nenhuma chave de embeddings é necessária. A interface usa navegação lateral por tarefa — consulta, ingestão e catálogo — para manter a tela principal focada no trabalho selecionado.

Na ingestão, o app segmenta documentos, calcula embeddings localmente e chama Claude para extrair entidades e relações. FastEmbed também calcula localmente o vetor da pergunta; a consulta faz busca vetorial por cosseno e duas formulações full-text (AND/OR), e o Reciprocal Rank Fusion combina as três listas. A expansão Cypher estática, parametrizada e allowlisted prioriza âncoras presentes na pergunta (com fallback aos seeds dos trechos) e percorre até dois saltos. O shortestPath escolhe uma âncora e tenta alcançar um endpoint a dois saltos no subgrafo, ou um vizinho conectado se não houver caminho mais longo disponível. Regras conservadoras podem complementar/reparar relações explícitas sem novas chamadas Claude. Claude recebe pergunta, trechos e relações recuperados para gerar a resposta.

## Escopo, dados e limites

`docs/examples/corpus/` contém 100 registros autorais e inteiramente fictícios: 50 filmes e 50 séries, distribuídos em quatro arquivos TXT. Cada parágrafo é segmentado como trecho separado. Ingerir o corpus completo faz aproximadamente 100 chamadas à API Claude; isso pode consumir cota ou gerar cobrança. O cálculo de embeddings é local.

A Anthropic recebe documentos para extração e pergunta/evidências para geração de respostas. Não há envio dos textos ao provedor de embeddings. O primeiro uso baixa o modelo do Hugging Face; os pesos ficam no volume persistente `app_data`. Depois do download, a inferência roda na CPU local. Planeje espaço em disco e memória conforme o equipamento.

## Estrutura

```text
CineGrafo/
├── cinegrafo/                 # cliente Claude, embeddings locais, Neo4j e recuperação
├── tests/                     # testes unitários
├── scripts/generate_demo_corpus.py # gerador reproduzível dos exemplos sintéticos
├── docs/
│   ├── README.md              # documentação técnica
│   ├── CLAUDE_API.md          # chave Anthropic e uso externo dos dados
│   ├── EMBEDDINGS.md          # FastEmbed local, Neo4j vetorial e backfill
│   ├── COMANDOS_WINDOWS.md    # PowerShell e Neo4j Desktop
│   ├── DEMO.md                # roteiro de demonstração
│   ├── ARCHITECTURE.mmd       # fonte editável do diagrama
│   ├── ARCHITECTURE.png       # diagrama renderizado
│   └── examples/corpus/       # documentos fictícios
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml         # app; sem Neo4j container
├── requirements.txt
└── .env.example
```

## Requisitos e execução

- Docker Desktop + Docker Compose v2 no Windows.
- Neo4j Desktop iniciado com Bolt acessível pelo Docker em `127.0.0.1:7687`.
- Chave Anthropic com acesso à Messages API e cota disponível.
- Internet na primeira inicialização para baixar o modelo MiniLM; inferência subsequente é local.

Configure `.env` com a senha Neo4j e `ANTHROPIC_API_KEY`. Nenhuma outra credencial é necessária: os vetores são locais. Depois execute:

```powershell
docker compose up --build -d --remove-orphans
docker compose ps
docker compose logs --tail 50 app
```

`--remove-orphans` pode remover containers órfãos de versões anteriores. O serviço atual é somente `app`; Neo4j Desktop permanece externo. O volume `app_data` guarda uploads e cache de pesos. Não use `docker compose down -v` se quiser preservar esse cache e dados do app.

Acesse `http://127.0.0.1:8501`. Na barra lateral, expanda **Neo4j** e clique **Preparar índices e constraints**; aguarde `ONLINE`. Em **Ingestão**, abra **Manutenção da base** para **Gerar embeddings faltantes** ou **Verificar relações existentes**. Esta última ação só cria relações entre entidades já ligadas ao chunk quando a citação literal contém ambas e corresponde a um padrão explícito; não reingere documentos nem chama Claude. A vetorização só grava `Chunk.embedding_local` e preserva fontes, textos, entidades e relações.

## Schema Neo4j

O botão de schema cria constraints de unicidade para IDs de fonte, chunk e entidade; índices range para `Entity.canonical` e `Chunk.created_at`; índice full-text `chunk_fulltext` sobre `Chunk.text`; e índice vetorial `chunk_embedding_local` sobre `Chunk.embedding_local`, com 384 dimensões e similaridade cosseno.

```cypher
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state
RETURN name, type, labelsOrTypes, properties, state;
```

Índices esperados incluem `chunk_fulltext`, `chunk_embedding_local`, `entity_canonical` e `chunk_created`; constraints incluem `source_id`, `chunk_id`, `entity_id`.

## Retrieval, geração e proveniência

FastEmbed calcula localmente vetores para chunks (`input_type=document`) e perguntas (`input_type=query`). O RRF combina três listas — vetorial, full-text AND e full-text OR — pela soma `1/(60+rank)`. A expansão Cypher prioriza entidades citadas na pergunta e usa relações de domínio allowlisted, somente leitura, até dois saltos. O `shortestPath` escolhe uma entidade-âncora e, quando o subgrafo recuperado permite, um destino a dois saltos; caso contrário, usa um vizinho conectado. A consulta apresenta primeiro a resposta e o subgrafo; trechos, proveniência e Cypher ficam em seções recolhíveis para inspeção. O caminho mínimo é auxiliar; não prova relevância. Confiança das relações permanece fixa em 0.5, sem calibração.

A extração Claude usa JSON Schema; após o parse, o app mantém até oito entidades e cinco relações por trecho. Valida nomes/tipos e exige evidência contígua presente no chunk com ambos os endpoints antes de gravar uma aresta. Regras determinísticas com a mesma validação recuperam relações explícitas omitidas pelo LLM. A resposta é solicitada em português, deve se apoiar no contexto recuperado e indicar trechos; o modelo ainda pode errar.

## Testes

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Testes unitários não comprovam acesso à API, execução integrada no Neo4j Desktop nem eficácia de retrieval. Conduza avaliação anotada antes de afirmar desempenho.

## Módulos

- `cinegrafo/graph.py`: ingestão, recuperação vetorial/full-text AND/OR, RRF, expansão e caminho mínimo.
- `cinegrafo/embeddings.py`: inferência local FastEmbed/ONNX Runtime e validação de 384 dimensões.
- `cinegrafo/llm.py`: extração estruturada e geração de resposta via Claude API.
- `cinegrafo/documents.py`: extração TXT/PDF/DOCX e chunking.
- `cinegrafo/schema.py`: constraints, índices full-text/range e índice vetorial cosine.
- `streamlit_app.py`: ingestão, consulta, evidências, subgrafo e grade de entidades.
