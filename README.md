# MongoDB RAG Agent - Busca Inteligente em Base de Conhecimento

Sistema RAG agêntico que combina MongoDB Atlas Vector Search com Pydantic AI para recuperação inteligente de documentos.

## Funcionalidades

- **Busca Híbrida**: Combina busca vetorial semântica com busca textual por palavras-chave usando Reciprocal Rank Fusion (RRF)
  - Implementação manual de RRF fornece a mesma qualidade do `$rankFusion` do MongoDB (que está em preview)
  - Execução concorrente para latência mínima
- **Ingestão Multi-Formato**: PDF, Word, PowerPoint, Excel, HTML, Markdown, Transcrição de áudio
- **Chunking Inteligente**: Docling HybridChunker preserva estrutura do documento e limites semânticos
- **CLI Conversacional**: Interface baseada em Rich com streaming em tempo real e visibilidade de chamadas de ferramentas
- **Suporte a Múltiplos LLMs**: OpenAI, OpenRouter, Ollama, Gemini
- **Custo Efetivo**: Funciona completamente no tier gratuito do MongoDB Atlas (M0)

## Pré-requisitos

- Python 3.10+
- Conta no MongoDB Atlas (**o tier gratuito M0 funciona perfeitamente!**)
- Chave API do provedor LLM (OpenAI, OpenRouter, etc.)
- Chave API do provedor de embeddings (OpenAI ou OpenRouter recomendado)
- Gerenciador de pacotes UV

## Início Rápido

### 1. Instalar o Gerenciador de Pacotes UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clonar e Configurar o Projeto

```bash
git clone https://github.com/coleam00/MongoDB-RAG-Agent.git
cd MongoDB-RAG-Agent

# Criar ambiente virtual e instalar dependências
uv venv
source .venv/bin/activate  # Unix/Mac
.venv\Scripts\activate     # Windows
uv sync
```

### 3. Configurar MongoDB Atlas

1. Acesse [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) e crie uma conta gratuita
2. Clique em **"Create"** → Escolha o tier **M0 Free** → Selecione a região → Clique em **"Create Deployment"**
3. O **Quickstart Wizard** aparece - configure a segurança:
   - **Database User**: Crie nome de usuário e senha (salve estes dados!)
   - **Network Access**: Clique em "Add My Current IP Address"
4. Clique em **"Connect"** → **"Drivers"** → Copie sua string de conexão
   - Formato: `mongodb+srv://username:<password>@cluster.mongodb.net/?appName=YourApp`
   - Substitua `<password>` pela sua senha real

**Nota**: O banco de dados (`rag_db`) e as coleções (`documents`, `chunks`) serão criados automaticamente quando você executar a ingestão no passo 6.

### 4. Configurar Variáveis de Ambiente

```bash
# Copiar o arquivo de exemplo
cp .env.example .env
```

Edite `.env` com suas credenciais:
- **MONGODB_URI**: String de conexão do passo 3
- **LLM_API_KEY**: Sua chave API do provedor LLM (OpenRouter, OpenAI, etc.)
- **EMBEDDING_API_KEY**: Sua chave API para embeddings (como OpenAI ou OpenRouter)

### 5. Validar Configuração

```bash
uv run python -m src.test_config
```

Você deve ver: `[OK] TODAS AS VERIFICAÇÕES DE CONFIGURAÇÃO PASSARAM`

### 6. Executar Pipeline de Ingestão

```bash
# Adicione seus documentos na pasta documents/
uv run python -m src.ingestion.ingest -d ./documents
```

Isso irá:
- Processar seus documentos (PDF, Word, PowerPoint, Excel, Markdown, etc.)
- Dividi-los em chunks de forma inteligente
- Gerar embeddings
- Armazenar tudo no MongoDB (`rag_db.documents` e `rag_db.chunks`)

### 7. Criar Índices de Busca no MongoDB Atlas

**Importante**: Crie estes índices APENAS DEPOIS de executar a ingestão - você precisa de dados na sua coleção `chunks` primeiro.

No MongoDB Atlas, vá para **Database** → **Search and Vector Search** → **Create Search Index**

**1. Índice de Busca Vetorial**
- Escolha: **"Vector Search"**
- Database: `rag_db`
- Collection: `chunks`
- Nome do índice: `vector_index`
- JSON:
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

**2. Índice Atlas Search**
- Clique em **"Create Search Index"** novamente
- Escolha: **"Atlas Search"**
- Database: `rag_db`
- Collection: `chunks`
- Nome do índice: `text_index`
- JSON:
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      }
    }
  }
}
```

Aguarde 1-5 minutos para ambos os índices serem construídos (status: "Building" → "Active").

### 8. Executar o Agente

```bash
uv run python -m src.cli
```

Agora você pode fazer perguntas e o agente irá buscar em sua base de conhecimento!

## Estrutura do Projeto

```
MongoDB-RAG-Agent/
├── src/                           # Implementação MongoDB (COMPLETA)
│   ├── settings.py               # ✅ Gerenciamento de configuração
│   ├── providers.py              # ✅ Provedores LLM/embedding
│   ├── dependencies.py           # ✅ Conexão MongoDB & AgentDependencies
│   ├── test_config.py            # ✅ Validação de configuração
│   ├── tools.py                  # ✅ Ferramentas de busca (semântica, textual, híbrida RRF)
│   ├── agent.py                  # ✅ Agente Pydantic AI com ferramentas de busca
│   ├── cli.py                    # ✅ CLI conversacional baseado em Rich
│   ├── prompts.py                # ✅ Prompts do sistema
│   └── ingestion/
│       ├── chunker.py            # ✅ Wrapper Docling HybridChunker
│       ├── embedder.py           # ✅ Geração de embeddings em lote
│       └── ingest.py             # ✅ Pipeline de ingestão MongoDB
├── examples/                      # Referência PostgreSQL (NÃO MODIFICAR)
│   ├── agent.py                  # Referência: Padrões de agente Pydantic AI
│   ├── tools.py                  # Referência: Ferramentas de busca PostgreSQL
│   └── cli.py                    # Referência: Interface CLI Rich
├── documents/                     # Pasta de documentos (13 documentos de exemplo incluídos)
├── .claude/                       # Documentação do projeto
│   ├── PRD.md                    # Requisitos do produto
│   └── reference/                # Padrões MongoDB/Docling/Agent
├── .agents/
│   ├── plans/                    # Planos de implementação (todas as fases)
│   └── analysis/                 # Análise técnica & decisões
├── comprehensive_e2e_test.py      # ✅ Validação E2E completa (10/10 passou)
└── pyproject.toml                # Configuração de pacote UV
```

## Stack Tecnológico

- **Banco de Dados**: MongoDB Atlas (Vector Search + Full-Text Search)
- **Framework de Agente**: Pydantic AI 0.1.0+
- **Processamento de Documentos**: Docling 2.14+ (PDF, Word, PowerPoint, Excel, Áudio)
- **Driver Assíncrono**: PyMongo 4.10+ com API async nativa
- **CLI**: Rich 13.9+ (formatação de terminal e streaming)
- **Gerenciador de Pacotes**: UV 0.5.0+ (gerenciamento rápido de dependências)

## Implementação de Busca Híbrida

Este projeto usa **Reciprocal Rank Fusion (RRF) manual** para combinar resultados de busca vetorial e textual, fornecendo a mesma qualidade do operador `$rankFusion` do MongoDB enquanto funciona no **tier gratuito M0** (já que $rankFusion está em preview e não está disponível no tier M0).

### Como Funciona

1. **Busca Semântica** (`$vectorSearch`): Encontra conteúdo conceitualmente similar usando embeddings vetoriais
2. **Busca Textual** (`$search`): Encontra correspondências de palavras-chave com correspondência difusa para erros de digitação
3. **Mesclagem RRF**: Combina resultados usando a fórmula: `RRF_score = Σ(1 / (60 + rank))`
   - Documentos que aparecem em ambas as buscas recebem pontuações combinadas mais altas
   - Desduplicação automática
   - Constante padrão k=60 (comprovadamente eficaz em vários conjuntos de dados)

### Performance

- **Latência**: ~350-600ms por consulta (ambas as buscas executam concorrentemente)
- **Precisão**: 100% de taxa de sucesso em testes de validação
- **Custo**: $0/mês (funciona no tier gratuito M0)

## Exemplos de Uso

### CLI Interativo

```bash
uv run python -m src.cli
```

**Exemplo de conversa:**
```
Você: Qual é a meta de receita da NeuralFlow AI para 2025?

  [Chamando ferramenta] search_knowledge_base
    Consulta: Meta de receita da NeuralFlow AI para 2025
    Tipo: hybrid
    Resultados: 5
  [Busca concluída com sucesso]
```
