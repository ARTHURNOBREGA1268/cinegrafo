# Comandos Windows — CineGrafo + Claude + embeddings locais + Neo4j Desktop

Execute no PowerShell, na raiz extraída do projeto (pasta que contém `docker-compose.yml`).

## 1. Configurar `.env`

Somente se ainda não existir:

```powershell
Copy-Item .env.example .env
notepad .env
```

Preencha `NEO4J_PASSWORD` e `ANTHROPIC_API_KEY`. Não há chave de embeddings: os vetores são calculados localmente pelo FastEmbed. Não substitua um `.env` existente pelo exemplo, para preservar sua senha e chave. Nunca compartilhe credenciais.

Claude recebe documentos para extração e pergunta/evidências para respostas. Textos não são enviados a um provedor externo de embeddings. O download inicial do modelo FastEmbed é feito do Hugging Face e seu cache fica no volume persistente do app.

## 2. Confirmar Neo4j Desktop

Inicie o banco `neo4j` no Neo4j Desktop:

```powershell
Test-NetConnection 127.0.0.1 -Port 7687
Get-NetTCPConnection -LocalPort 7687 -State Listen -ErrorAction SilentlyContinue |
  Format-Table LocalAddress, LocalPort, OwningProcess
```

O TCP deve retornar `True`. Se o container não alcançar o Bolt, configure `server.default_listen_address=0.0.0.0` no `neo4j.conf` do DBMS Desktop e reinicie pelo Desktop. Não exponha a porta publicamente.

## 3. Atualizar o app

```powershell
docker compose up --build -d --force-recreate --remove-orphans app
docker compose ps
docker compose logs --tail 60 app
```

O app continua conectando ao Neo4j Desktop no host. Não use `docker compose down -v`.

Confirme sem imprimir a chave:

```powershell
docker compose exec app python -c "from cinegrafo.config import settings; print('Claude configurado:', bool(settings.anthropic_api_key)); print('Cache local:', settings.embedding_cache_dir); print('Threads:', settings.embedding_threads)"
```

## 4. Criar índice local e preencher embeddings existentes

1. Abra `http://127.0.0.1:8501`.
2. Na navegação lateral, expanda **Neo4j** e clique **Preparar índices e constraints**; aguarde `chunk_embedding_local` ficar `ONLINE`. É um índice de 384 dimensões em `Chunk.embedding_local`.
3. Selecione **Ingestão**, abra **Manutenção da base** e clique **Gerar embeddings faltantes**. Na primeira vez, o FastEmbed baixa o modelo; aguarde a conclusão. A aplicação atualiza somente `embedding_local`, preservando todos os nós, relações e textos já existentes.
4. A rotina ignora trechos já vetorizados, então pode ser retomada após uma falha. Documentos novos recebem vetores automaticamente.

O modelo e seu cache ficam em `/app/data/embedding_models`, dentro do volume persistente `app_data`. A primeira execução exige internet e espaço em disco; a inferência ocorre na CPU local.

## 5. Diagnóstico Neo4j

```cypher
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state
WHERE name IN ['chunk_embedding_local', 'chunk_fulltext']
RETURN name, type, labelsOrTypes, properties, state;

MATCH (c:Chunk)
RETURN count(c) AS trechos,
       count(c.embedding_local) AS vetorizados_locais;
```

Se alterou `.env`, recrie o app: `docker compose up -d --force-recreate app`. Erros Claude dependem da chave/cota Anthropic; falha ao baixar o modelo exige verificar internet, DNS e espaço em disco. Falhas Neo4j exigem verificar o Neo4j Desktop/Bolt.

## 6. Parar e iniciar sem apagar dados

```powershell
docker compose stop
docker compose start
```
