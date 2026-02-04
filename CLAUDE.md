# RAG Agent Development Instructions (Chroma + Projects)

## Project Overview

Agentic RAG system with Chroma as vector store and projects defined in `projects.json`. Uses Docling for multi-format ingestion, Pydantic AI for the agent, and semantic search always filtered by `project_id` (multi-project via tag). Built with UV and type-safe Pydantic models.

## Core Principles

1. **TYPE SAFETY IS NON-NEGOTIABLE**
   - All functions, methods, and variables MUST have type annotations
   - Use Pydantic models for all data structures (chunks, search results)
   - No `Any` types without explicit justification

2. **KISS** (Keep It Simple, Stupid)
   - Prefer simple, readable solutions over clever abstractions
   - Don't build fallback mechanisms unless absolutely necessary
   - One Chroma collection `rag_chunks` for all projects

3. **YAGNI** (You Aren't Gonna Need It)
   - Don't build features until they're actually needed
   - MVP first, enhancements later

4. **ASYNC ALL THE WAY**
   - All I/O operations MUST be async (embeddings, LLM calls); Chroma calls via `asyncio.to_thread`
   - Proper cleanup with `try/finally` or context managers

**Architecture:**

```
src/
├── agent.py           # Pydantic AI agent (search by project_id or tag)
├── cli.py             # Rich-based conversational CLI
├── dependencies.py    # Chroma client + collection, OpenAI embeddings
├── projects.json      # Project definitions (not in Chroma)
├── projects.py        # resolve_project, get_projects_by_tag
├── tools.py           # semantic_search (where project_id), multi_project_search
├── prompts.py         # System prompts
└── ingestion/
    ├── chunker.py     # Docling HybridChunker wrapper
    ├── embedder.py    # Batch embedding generation
    └── ingest.py      # Pipeline: resolve_project → chunk → embed → Chroma add
```

---

## Documentation Style

**Use Google-style docstrings** for all functions, classes, and modules:

```python
async def semantic_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    match_count: Optional[int] = None
) -> list[SearchResult]:
    """
    Perform pure semantic search using vector similarity.

    Args:
        ctx: Agent runtime context with dependencies
        query: Search query text
        match_count: Number of results to return (default: 10)

    Returns:
        List of search results ordered by similarity

    Raises:
        ValueError: If match_count exceeds maximum allowed
    """
```

---

## Development Workflow

**Setup environment:**
```bash
# Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

**Run ingestion:**
```bash
# Documents must be under documents/<project_key>/ (project_key from projects.json)
uv run python -m src.ingestion.ingest -d ./documents

# With options
uv run python -m src.ingestion.ingest -d ./documents --chunk-size 1000 --no-clean
```

**Run CLI agent:**
```bash
uv run python -m src.cli
```

**Common CLI commands:**
- `info` - Show system configuration
- `clear` - Clear screen
- `exit` / `quit` / `q` - Exit agent

---

## Configuration Management

### Environment Variables

**ALL configuration in .env file:**
```bash
# Chroma (vector store)
CHROMA_PATH=./chroma_data

# LLM Provider
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=anthropic/claude-haiku-4.5
LLM_BASE_URL=https://openrouter.ai/api/v1

# Embedding Provider
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

### Pydantic Settings

**Use Pydantic Settings for type-safe configuration:**
```python
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    chroma_path: str = Field(default="./chroma_data", description="Chroma persistent path")
    llm_api_key: str = Field(..., description="LLM provider API key")
    embedding_model: str = Field(default="text-embedding-3-small")
```

---

## Error Handling

### General Pattern

```python
try:
    result = await operation()
except SpecificError as e:
    logger.exception("operation_failed", context="value", error=str(e))
    raise
```

### Chroma Operations

```python
# Chroma API is sync; run in thread to avoid blocking event loop
result = await asyncio.to_thread(
    collection.query,
    query_embeddings=[embedding],
    n_results=10,
    where={"project_id": project_id},
    include=["documents", "metadatas", "distances"],
)
```

### API Calls (Embeddings, LLM)

```python
from openai import APIError, RateLimitError

try:
    result = await client.api_call(params)
except RateLimitError as e:
    logger.warning("api_rate_limited", retry_after=e.retry_after)
    await asyncio.sleep(e.retry_after or 5)
    # Retry logic here
except APIError as e:
    logger.exception("api_error", status_code=e.status_code)
    raise
```

### Document Processing

```python
try:
    result = converter.convert(file_path)
except Exception as e:
    logger.exception(
        "document_conversion_failed",
        file=file_path,
        format=os.path.splitext(file_path)[1]
    )
    # Continue processing other documents, don't crash pipeline
    return None
```

---

## Testing

**Tests mirror the examples directory structure:**

```
examples/agent.py        →  tests/test_agent.py
examples/tools.py        →  tests/test_tools.py
examples/ingestion/      →  tests/ingestion/
```

### Unit Tests

```python
import pytest
from examples.ingestion.chunker import DoclingHybridChunker, ChunkingConfig

@pytest.mark.unit
async def test_chunker_creates_valid_chunks():
    """Test that chunker creates properly formatted chunks."""
    config = ChunkingConfig(max_tokens=512)
    chunker = DoclingHybridChunker(config)

    content = "# Heading\n\nSome content here..."
    chunks = await chunker.chunk_document(
        content=content,
        title="Test Doc",
        source="test.md"
    )

    assert len(chunks) > 0
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
```

### Integration Tests

```python
@pytest.mark.integration
async def test_chroma_semantic_search(deps):
    """Test semantic search against Chroma (always with where project_id)."""
    results = await semantic_search(
        ctx=test_context,
        query="test",
        project_id="master_detox",
        match_count=5,
    )
    assert all(r.metadata.get("project_id") == "master_detox" for r in results)
```

**Run tests:**
```bash
uv run pytest tests/ -v

# Run specific markers
uv run pytest tests/ -m unit
uv run pytest tests/ -m integration
```

---

## Common Pitfalls

### 1. Query without where
```python
# ❌ WRONG - Never query Chroma without project filter by default
result = collection.query(query_embeddings=[emb], n_results=10)

# ✅ CORRECT - Always filter by project_id
result = collection.query(
    query_embeddings=[emb],
    n_results=10,
    where={"project_id": project_id},
)
```

### 2. Project resolution after embedding
```python
# ❌ WRONG - Resolve project after generating embeddings
chunks = embed_chunks(chunks)
project = resolve_project(project_key)

# ✅ CORRECT - Resolve project before embedding (required for ids and metadata)
project = resolve_project(project_key)
# ... then chunk and embed
```

### 3. Missing DoclingDocument for HybridChunker
```python
# ❌ WRONG - Passing raw text to HybridChunker
chunks = chunker.chunk(dl_doc=markdown_text)

# ✅ CORRECT - Pass DoclingDocument from converter
result = converter.convert(file_path)
chunks = chunker.chunk(dl_doc=result.document)
```

### 4. Chroma sync API blocking event loop
```python
# ❌ WRONG - Chroma is sync; calling directly blocks
collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

# ✅ CORRECT - Run in thread
await asyncio.to_thread(
    collection.add,
    ids=ids,
    documents=docs,
    embeddings=embs,
    metadatas=metas,
)
```

### 5. Query without where (project_id)
```python
# ❌ WRONG - Never query without project filter by default
result = collection.query(query_embeddings=[emb], n_results=10)

# ✅ CORRECT - Always filter by project_id
result = collection.query(
    query_embeddings=[emb],
    n_results=10,
    where={"project_id": project_id},
)
```

---

## Quick Reference

**Chroma Operations:**
```python
# One collection for all projects
collection = client.get_or_create_collection("rag_chunks", metadata={...})

# Add chunks (deterministic id: project_id:doc_slug:chunk_index)
await asyncio.to_thread(
    collection.add,
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"project_id": pid, "project_name": name, "document_title": title, ...}],
)

# Query (always with where project_id)
result = await asyncio.to_thread(
    collection.query,
    query_embeddings=[query_embedding],
    n_results=10,
    where={"project_id": project_id},
    include=["documents", "metadatas", "distances"],
)
```

**Embedding Generation:**
```python
# Single
embedding = await client.embeddings.create(model=model, input=text)

# Batch (ALWAYS prefer batching)
embeddings = await client.embeddings.create(model=model, input=texts)
```

**Docling Conversion:**
```python
# Convert any supported format
result = converter.convert(file_path)
markdown = result.document.export_to_markdown()
docling_doc = result.document  # Keep for HybridChunker

# Chunk with context preservation
chunks = list(chunker.chunk(dl_doc=docling_doc))
```

**Pydantic AI Agent:**
```python
# Define agent with StateDeps
agent = Agent(model, deps_type=StateDeps[State], system_prompt=prompt)

# Add tool
@agent.tool
async def tool_func(ctx: RunContext[StateDeps[State]], arg: str) -> str:
    """Tool description."""
    pass

# Run with streaming
async with agent.iter(input, deps=deps, message_history=history) as run:
    async for node in run:
        # Handle nodes (see .claude/reference/agent-tools.md)
        pass
```

---

## Implementation-Specific References

For detailed implementation patterns, see:

- **Docling ingestion**: `.claude/reference/docling-ingestion.md`
  - Document conversion for all formats
  - HybridChunker usage and configuration
  - Audio transcription with Whisper ASR

- **Agent & tools**: `.claude/reference/agent-tools.md`
  - Pydantic AI agent patterns
  - Tool definitions and best practices
  - Streaming implementation details

Chroma: one collection `rag_chunks`; project_id resolved before embedding; IDs deterministic; query always with `where`.
