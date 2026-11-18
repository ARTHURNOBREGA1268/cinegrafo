# 10Pila

Protótipo acadêmico GraphRAG/KAG para filmes e séries: Claude API para ingestão Text-to-Graph e respostas fundamentadas; embeddings multilíngues locais via FastEmbed/ONNX Runtime; Neo4j LPG para busca vetorial cosseno, full-text e expansão Cypher multi-hop.

## Começar

1. Copie `.env.example` para `.env` somente se ainda não existir; configure a senha do Neo4j Desktop e `ANTHROPIC_API_KEY`. Não é necessária chave de embeddings.
2. Inicie o Neo4j Desktop no Windows e confirme o Bolt em `127.0.0.1:7687` acessível pelo Docker.
3. Execute `docker compose up --build -d --remove-orphans`.
4. Abra `http://127.0.0.1:8501`.
5. Na barra lateral, expanda **Neo4j** e clique **Preparar índices e constraints**; aguarde ficarem `ONLINE`.
6. Para os trechos já existentes no Neo4j, selecione **Ingestão** na navegação lateral, abra **Manutenção da base** e clique **Gerar embeddings faltantes**. Isso grava `Chunk.embedding_local` sem alterar fontes, textos, entidades ou relações já existentes.
7. Na primeira vetorização, o modelo será baixado do Hugging Face; aguarde. O cache fica no volume persistente `app_data`. Documentos novos são vetorizados localmente durante a ingestão.

O Compose usa `name: 10pila`, o serviço `app` e o volume `app_data`; não cria Neo4j nem serviços adicionais. O Neo4j Desktop do Windows continua sendo o banco local. Ao mudar do nome anterior `jurisgraph`, o Docker criará um container e um volume de cache `10pila_app_data` novos; o volume antigo não é apagado e o banco Neo4j Desktop externo não é afetado. Não use `docker compose down -v` se quiser preservar volumes locais.

## Dados e privacidade

Claude recebe os documentos para extração e as perguntas/evidências para resposta. Embeddings são calculados localmente; o texto não é enviado a nenhum provedor externo de embeddings. O download inicial dos pesos MiniLM vem do Hugging Face e fica em cache local. A API Claude pode consumir cota ou gerar cobrança; use apenas documentos autorizados e mantenha sua chave no `.env` local.

O modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` gera vetores de 384 dimensões. O índice `chunk_embedding_local` usa cosseno. A busca semântica é combinada por RRF com full-text AND/OR, mas isso não prova a correção das respostas; continua necessária avaliação empírica anotada.

## Documentação

- Setup da Anthropic: [`docs/CLAUDE_API.md`](docs/CLAUDE_API.md)
- Embeddings locais, índice e migração não destrutiva: [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md)
- Comandos Windows e Neo4j Desktop: [`docs/COMANDOS_WINDOWS.md`](docs/COMANDOS_WINDOWS.md)
- Documentação técnica: [`docs/README.md`](docs/README.md)
- Corpus de demonstração fictício: [`docs/examples/corpus/`](docs/examples/corpus/)

## Testes

```powershell
python -m pytest -q
```

Os testes unitários não substituem validação integrada com chave Anthropic, download inicial do modelo e Neo4j Desktop.
