# Embeddings locais com FastEmbed e Neo4j

## Arquitetura

O CineGrafo usa a API Claude para extração de entidades/relações e redação de respostas. Os embeddings são calculados dentro do container do aplicativo usando FastEmbed/ONNX Runtime e o modelo multilíngue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, com vetores de 384 dimensões. A busca grava os vetores em `Chunk.embedding_local`; o índice Neo4j `chunk_embedding_local` usa similaridade cosseno. O retriever combina rankings vetorial, full-text AND e full-text OR por Reciprocal Rank Fusion (RRF).

O modelo é livre para baixar e executar localmente, sem chave nem chamada a um serviço de embeddings. Na primeira execução que precisar de vetores, o app baixa os pesos do Hugging Face; depois, os arquivos permanecem no volume `app_data` em `/app/data/embedding_models`. A primeira inicialização requer internet e pode levar mais tempo. A inferência subsequente é local e utiliza CPU do computador.

## Privacidade e custos

Os textos usados para embeddings não são enviados a nenhum serviço externo de embeddings. A API Anthropic continua recebendo documentos para extração e as perguntas/evidências necessárias para gerar respostas. Portanto, use somente conteúdo autorizado para envio à Claude e verifique cota/custo da conta Anthropic. O modelo de embedding não exige chave externa.

## Revetorizar trechos existentes sem apagar o grafo

1. Atualize o código e reconstrua somente o app:

   ```powershell
   docker compose up --build -d --force-recreate app
   ```

2. Abra a aplicação, expanda **Neo4j** na barra lateral, clique **Preparar índices e constraints** e aguarde `chunk_embedding_local` ficar `ONLINE`.
3. Selecione **Ingestão**, abra **Manutenção da base** e clique **Gerar embeddings faltantes**. O processamento é em lotes retomáveis e grava somente `Chunk.embedding_local`.
4. Acompanhe o progresso até terminar. Não apague banco, chunks, fontes, entidades ou relações. Documentos novos receberão embeddings locais automaticamente.

Trechos gravados por uma versão anterior podem já possuir vetores. A rotina de revetorização grava apenas `Chunk.embedding_local`; nenhum nó, relação ou texto é apagado. O código atual consulta exclusivamente a propriedade e o índice locais.

## Diagnóstico Neo4j

```cypher
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state
WHERE name IN ['chunk_embedding_local', 'chunk_fulltext']
RETURN name, type, labelsOrTypes, properties, state;

MATCH (c:Chunk)
RETURN count(c) AS trechos,
       count(c.embedding_local) AS trechos_com_embedding_local;
```

## Limites

O model card declara um espaço denso de 384 dimensões e suporte multilíngue. Isso não garante a mesma qualidade de outro modelo, nem desempenho específico em português; será necessário avaliar com perguntas e referências anotadas. O modelo limita o contexto de entrada; mantenha os chunks curtos, como no corpus do projeto. RRF combina posições, não scores vetoriais e lexicais brutos. A confiança das relações permanece demonstrativa e não calibrada.

## Referências

- FastEmbed: https://qdrant.github.io/fastembed/
- Lista de modelos FastEmbed suportados: https://qdrant.github.io/fastembed/examples/Supported_Models/
- Model card MiniLM multilíngue: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Índices vetoriais Neo4j: https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/vector-indexes/
- Sentence-BERT: https://arxiv.org/abs/1908.10084
