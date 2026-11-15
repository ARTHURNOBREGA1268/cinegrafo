# Roteiro da demonstração

## Preparação

1. Inicie Neo4j Desktop e confirme que o banco `neo4j` está ativo.
2. Configure `ANTHROPIC_API_KEY` no `.env`; preserve a senha Neo4j. Não é necessária chave de embeddings; os vetores são calculados localmente.
3. Execute `docker compose up --build -d --remove-orphans` e confirme o serviço `app`.
4. Acesse `http://127.0.0.1:8501`.
5. Na barra lateral, expanda **Neo4j**, clique **Preparar índices e constraints** e aguarde `ONLINE`, inclusive o índice vetorial local `chunk_embedding_local`.
6. Para dados que já existiam, selecione **Ingestão** na navegação lateral e abra **Manutenção da base**. Clique **Gerar embeddings faltantes**. Na primeira execução, o modelo é baixado; a rotina vetorializa somente chunks sem `embedding_local` e preserva conteúdo e relações.

## Fluxo a apresentar

1. O corpus tem 100 obras sintéticas em quatro TXT; cada parágrafo vira um trecho. Antes de ingerir, considere que a extração executa aproximadamente 100 chamadas Claude, que podem consumir cota ou gerar cobrança. Embeddings MiniLM rodam localmente. Para a apresentação, ingira apenas o documento necessário ou use dados previamente carregados.
2. Selecione **Consulta** na navegação lateral e pergunte:

> Quem atua nas duas obras da franquia Crônicas de Nébula, quais personagens interpreta e quais gêneros são associados a cada obra?

3. Mostre a resposta e o subgrafo. Abra **Evidências textuais**, **Relações, nós e proveniência** e **Caminho mínimo e detalhes da recuperação** para apresentar trechos, fontes, endpoint do caminho, full-text AND/OR, fusão RRF e Cypher executado. O shortestPath procura um destino a até dois saltos quando o subgrafo o permite.
4. Confirme os índices e os chunks com embedding no Neo4j Browser.

## Limites e transparência

- Documentos usados na extração/resposta são enviados à Anthropic. Embeddings são processados localmente; não há envio a serviço externo de embeddings. Não use materiais sem autorização para envio externo à Claude.
- A recuperação combina vetores locais MiniLM (cosseno, 384 dimensões) e busca lexical. Isso não garante precisão da resposta.
- Corpus sintético é inventado e não comprova fatos sobre filmes reais.
- Confiança 0,5 é constante demonstrativa, não probabilidade calibrada.
- Testes unitários não são avaliação de eficácia; não alegue métricas sem experimento anotado.

## Contingência

Se uma API estiver indisponível ou sem cota, não invente o resultado. Mostre componentes já preparados, explique a dependência externa e use evidências previamente produzidas apenas se forem reais e identificadas como tal. Nunca exponha as chaves em capturas ou logs compartilhados.
