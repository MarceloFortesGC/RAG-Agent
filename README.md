# RAG Agent - Busca por Projeto (Chroma)

Sistema RAG agêntico com Chroma como store vetorial, projetos definidos em `projects.json` e busca sempre filtrada por projeto (com opção multi-projeto por tag).

## Funcionalidades

- **Chroma**: Uma collection `rag_chunks` para todos os projetos; persistência em `./chroma_data`
- **Projetos**: `src/projects.json` define projetos (project_id, name, tags, domain, language); resolução por pasta na ingestão
- **Busca semântica**: Sempre com `where project_id`; multi-projeto por tag (ex.: tag "flutter")
- **Ingestão**: Estrutura `documents/<project_key>/` (ex.: `documents/master_detox/termos.pdf`); IDs determinísticos `project_id:doc_slug:chunk_index`
- **CLI**: Rich, streaming e ferramentas de busca (project_id ou tag)
- **Docling**: PDF, Word, PowerPoint, Excel, Markdown, áudio (Whisper)

## Pré-requisitos

- Python 3.10+
- UV
- Chaves: LLM (OpenRouter/OpenAI) e Embedding (OpenAI recomendado)

## Início Rápido

### 1. UV e dependências

```bash
uv venv
source .venv/bin/activate   # Unix/Mac
uv sync
```

### 2. Configuração

```bash
cp .env.example .env
```

Edite `.env`: `LLM_API_KEY`, `EMBEDDING_API_KEY`. Opcional: `CHROMA_PATH` (default `./chroma_data`).

### 3. Projetos

Projetos em `src/projects.json`. Chave = nome da pasta em `documents/` (ex.: `master_detox`, `qualitare`).

### 4. Documentos

Coloque arquivos em pastas por projeto:

```
documents/
  master_detox/
    termos_de_uso.pdf
    readme.md
  qualitare/
    manual.pdf
```

### 5. Validar e ingerir

```bash
uv run python -m src.test_config
uv run python -m src.ingestion.ingest -d ./documents
```

### 6. CLI

```bash
uv run python -m src.cli
```

- Busca em um projeto: a ferramenta usa o primeiro projeto de `projects.json` ou você informa `project_id`
- Busca em vários projetos (ex.: Flutter): use `tag="flutter"` na ferramenta

## Estrutura do Projeto

```
src/
  agent.py          # Agente Pydantic AI (busca por project_id ou tag)
  dependencies.py   # Chroma + OpenAI embeddings
  projects.json     # Definição de projetos (não vai para o Chroma)
  projects.py       # resolve_project, get_projects_by_tag
  tools.py          # semantic_search, multi_project_search, build_rag_context
  ingestion/
    ingest.py       # Pipeline: resolve_project → chunk → embed → Chroma add
```

## Checklist

- Mongo removido; uma collection Chroma `rag_chunks`
- `project_id` resolvido antes do embedding na ingestão
- IDs determinísticos `{project_id}:{document_slug}:{chunk_index}`
- Query sempre com `where` (por projeto); global via orquestração por tag
