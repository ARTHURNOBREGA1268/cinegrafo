# Claude API no 10Pila

## Papel e dados enviados

Claude é usado para extração Text-to-Graph e redação das respostas fundamentadas. O aplicativo usa embeddings locais via FastEmbed/ONNX Runtime; textos não são enviados a serviço remoto de embeddings. Na primeira execução, o modelo aberto é baixado do Hugging Face e armazenado no volume local `app_data`. Neo4j Desktop armazena o grafo e vetores.

Documentos selecionados para ingestão e perguntas/contexto recuperado para resposta são enviados à Anthropic. Não envie material pessoal, confidencial, protegido ou sem autorização. A API Anthropic está sujeita a disponibilidade, cota e cobrança; consulte a console. Reiniciar ou reconstruir containers não restaura cota esgotada.

## Criar e configurar a chave

1. Crie uma chave na [Anthropic Console](https://platform.claude.com/settings/keys).
2. Na pasta do projeto, copie `.env.example` para `.env` somente se ainda não existir.
3. Preencha `ANTHROPIC_API_KEY`; preserve a senha `NEO4J_PASSWORD` já configurada. Nenhuma outra chave é necessária: os embeddings rodam localmente. O exemplo usa `ANTHROPIC_MODEL=claude-sonnet-5-5`; se não estiver habilitado para a conta, use um model ID disponível na Console.
4. Recrie o app para carregar o `.env` atualizado:

```powershell
docker compose up --build -d --force-recreate --remove-orphans app
```

Confirme configuração sem imprimir segredos:

```powershell
docker compose exec app python -c "from cinegrafo.config import settings; print('Anthropic configurada:', bool(settings.anthropic_api_key)); print('Embeddings locais:', settings.embedding_cache_dir)"
```

## Segurança

A chave é carregada somente no servidor Streamlit. Não a cole em conversas, capturas, documentos ou código. O `.env` é ignorado pelo Git e excluído dos ZIPs. Se suspeitar de vazamento, revogue a chave e gere outra na Console.

Para testar Claude (esta chamada pode consumir cota):

```powershell
docker compose exec app python -c "from cinegrafo.llm import Claude; print(Claude().generate('Responda apenas: Claude conectado.', max_tokens=32))"
```

## Referências oficiais

- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages)
- [Autenticação Anthropic](https://platform.claude.com/docs/en/manage-claude/authentication)
- [API keys Anthropic](https://platform.claude.com/settings/keys)
- [FastEmbed](https://qdrant.github.io/fastembed/)
- [Modelo MiniLM multilíngue](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
