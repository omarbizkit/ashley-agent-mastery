# Dependency Configuration Specifications

## Hybrid RAG Knowledge Graph Agent

This document specifies the dependency configuration for the Hybrid RAG Knowledge Graph Agent. The implementation phase will convert these specifications to Python code.

---

## 1. Settings Specification (settings.py)

### Purpose
Environment configuration using pydantic-settings with python-dotenv for loading `.env` files.

### Required Environment Variables

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `LLM_API_KEY` | str | Yes | - | API key for the LLM provider (OpenAI-compatible) |
| `LLM_MODEL` | str | No | `gpt-4` | Model name to use for agent responses |
| `LLM_BASE_URL` | str | No | `https://api.openai.com/v1` | Base URL for the LLM API |
| `DATABASE_URL` | str | Yes | - | PostgreSQL connection URL (Supabase) |
| `EMBEDDING_API_KEY` | str | Yes | - | API key for embeddings |
| `EMBEDDING_MODEL` | str | No | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_BASE_URL` | str | No | `https://api.openai.com/v1` | Embedding API base URL |
| `NEO4J_URI` | str | Yes | - | Neo4j Aura connection URI |
| `NEO4J_USER` | str | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | str | Yes | - | Neo4j password |
| `USE_GRAPH_SEARCH` | bool | No | `true` | Enable/disable graph search |
| `LOG_LEVEL` | str | No | `INFO` | Logging level |

### Settings Class Structure

```
Settings (BaseSettings)
    model_config: ConfigDict
        - env_file = ".env"
        - env_file_encoding = "utf-8"
        - case_sensitive = False
        - extra = "ignore"

    Fields:
        llm_api_key: str (required)
        llm_model: str = "gpt-4"
        llm_base_url: str = "https://api.openai.com/v1"
        database_url: str (required)
        embedding_api_key: str (required)
        embedding_model: str = "text-embedding-3-small"
        embedding_base_url: str = "https://api.openai.com/v1"
        neo4j_uri: str (required)
        neo4j_user: str = "neo4j"
        neo4j_password: str (required)
        use_graph_search: bool = True
        log_level: str = "INFO"
```

### Helper Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `load_settings()` | `Settings` | Load settings with error handling |

### Global Instance
- `settings = load_settings()` - Single global settings instance

---

## 2. Providers Specification (providers.py)

### Purpose
Model provider configuration for LLM and embedding clients.

### Required Functions

#### get_llm_model()

```
Function: get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel

Parameters:
    model_choice: Optional override for model name

Returns:
    OpenAIModel configured with provider

Implementation:
    1. Use model_choice or settings.llm_model
    2. Create OpenAIProvider with settings.llm_base_url and settings.llm_api_key
    3. Return OpenAIModel(model_name, provider=provider)
```

#### get_embedding_client()

```
Function: get_embedding_client() -> openai.AsyncOpenAI

Returns:
    AsyncOpenAI client for embeddings

Implementation:
    1. Return openai.AsyncOpenAI(
           base_url=settings.embedding_base_url,
           api_key=settings.embedding_api_key
       )
```

### Required Imports

```
from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from .settings import settings
```

---

## 3. Dependencies Specification (dependencies.py)

### Purpose
Agent runtime dependencies using simple dataclass pattern.

### SearchPreferences Dataclass

```
@dataclass
class SearchPreferences:
    """Configuration for search behavior."""

    Fields:
        use_vector: bool = True      # Enable vector search
        use_graph: bool = True       # Enable graph search
        default_limit: int = 10      # Default result limit
```

### HybridRAGDependencies Dataclass

```
@dataclass
class HybridRAGDependencies:
    """Dependencies for the Hybrid RAG Knowledge Graph Agent."""

    Fields:
        embedding_client: openai.AsyncOpenAI    # Client for generating embeddings
        embedding_model: str = "text-embedding-3-small"  # Model for embeddings
        search_preferences: SearchPreferences = field(default_factory=SearchPreferences)
        session_id: Optional[str] = None        # Optional session tracking
```

### Required Imports

```
from dataclasses import dataclass, field
from typing import Optional
import openai
```

### Usage Pattern

```python
# Create dependencies for agent.run()
embedding_client = openai.AsyncOpenAI(
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key
)
deps = HybridRAGDependencies(
    embedding_client=embedding_client,
    embedding_model=settings.embedding_model,
    search_preferences=SearchPreferences()
)

# Use with agent
result = await agent.run("query", deps=deps)
```

---

## 4. Python Package Dependencies (pyproject.toml)

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic-ai` | `>=0.0.49` | Agent framework |
| `pydantic-settings` | `>=2.0.0` | Settings management |
| `asyncpg` | `>=0.29.0` | PostgreSQL async driver |
| `graphiti-core` | `>=0.5.0` | Knowledge graph integration |
| `neo4j` | `>=5.0.0` | Neo4j driver |
| `openai` | `>=1.0.0` | OpenAI client for embeddings |
| `python-dotenv` | `>=1.0.0` | Environment file loading |
| `rich` | `>=13.0.0` | CLI formatting and streaming |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `>=8.0.0` | Testing framework |
| `pytest-asyncio` | `>=0.23.0` | Async test support |
| `pytest-cov` | `>=4.0.0` | Coverage reporting |
| `mypy` | `>=1.0.0` | Type checking |
| `ruff` | `>=0.1.0` | Linting |

### pyproject.toml Structure

```toml
[project]
name = "hybrid-rag-agent"
version = "0.1.0"
description = "Hybrid RAG Knowledge Graph Agent with Pydantic AI"
requires-python = ">=3.11"
dependencies = [
    "pydantic-ai>=0.0.49",
    "pydantic-settings>=2.0.0",
    "asyncpg>=0.29.0",
    "graphiti-core>=0.5.0",
    "neo4j>=5.0.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
```

---

## 5. Environment Template (.env.example)

```bash
# =============================================================================
# LLM Configuration (OpenAI-compatible)
# =============================================================================
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4
LLM_BASE_URL=https://api.openai.com/v1

# =============================================================================
# Database Configuration (Supabase PostgreSQL)
# =============================================================================
# Get from: Supabase Dashboard > Project Settings > Database > Connection String > URI
DATABASE_URL=postgresql://postgres.xxxxxxxxxxxx:your-password@aws-0-us-east-1.pooler.supabase.com:6543/postgres

# =============================================================================
# Embedding Configuration
# =============================================================================
EMBEDDING_API_KEY=sk-your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1

# =============================================================================
# Neo4j Configuration (Neo4j Aura Free)
# =============================================================================
# Get from: Neo4j Aura Console after creating your free instance
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# =============================================================================
# Feature Flags
# =============================================================================
USE_GRAPH_SEARCH=true

# =============================================================================
# Application Settings
# =============================================================================
LOG_LEVEL=INFO
```

---

## 6. File Structure

```
hybrid-rag-agent/
├── settings.py           # Environment configuration (BaseSettings)
├── providers.py          # Model provider setup (get_llm_model, get_embedding_client)
├── dependencies.py       # Agent dependencies (HybridRAGDependencies dataclass)
├── agent.py              # Agent initialization (uses all above)
├── cli.py                # CLI interface (uses dependencies)
├── pyproject.toml        # Package dependencies
├── .env.example          # Environment template
├── .env                  # Actual environment (git-ignored)
├── utils/
│   ├── __init__.py       # Exports all utilities
│   ├── db_utils.py       # Database connection and search (copied from rag_pipeline)
│   ├── graph_utils.py    # Graphiti knowledge graph (copied from rag_pipeline)
│   ├── models.py         # Pydantic models (copied from rag_pipeline)
│   └── providers.py      # Provider utilities (optional, main providers.py is root)
├── sql/
│   └── schema.sql        # Database schema (copied from rag_pipeline)
└── tests/
    ├── __init__.py
    └── test_agent.py     # Agent tests with TestModel
```

---

## 7. Testing Configuration

### Test Dependencies Fixture

```python
@pytest.fixture
def mock_embedding_client():
    """Create mock embedding client for tests."""
    client = AsyncMock()
    client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    return client

@pytest.fixture
def test_dependencies(mock_embedding_client):
    """Create test dependencies."""
    return HybridRAGDependencies(
        embedding_client=mock_embedding_client,
        embedding_model="text-embedding-3-small",
        search_preferences=SearchPreferences()
    )
```

### Agent Testing Pattern

```python
from pydantic_ai.models.test import TestModel

def test_agent_basic(test_dependencies):
    test_model = TestModel()
    with agent.override(model=test_model):
        result = agent.run_sync("Test query", deps=test_dependencies)
        assert result.output is not None
```

---

## 8. Key Design Decisions

### Simplicity Principles Applied

1. **Single LLM Provider**: OpenAI-compatible only (no provider switching logic)
2. **Minimal Settings**: Only essential environment variables
3. **Simple Dataclass**: HybridRAGDependencies is a basic dataclass, not a complex class
4. **No Factory Patterns**: Direct instantiation of dependencies
5. **No Fallback Models**: Single model configuration (keep simple)

### Why These Choices

| Decision | Rationale |
|----------|-----------|
| BaseSettings over plain env | Automatic validation, type coercion, .env file support |
| Dataclass over Pydantic model | Simpler for runtime dependencies, no validation overhead |
| Single providers.py file | All provider logic in one place, easy to understand |
| Global settings instance | Avoid repeated loading, consistent configuration |
| Embedding client in deps | Allows mocking in tests, type-safe context access |

---

## 9. Integration Points

### With Agent (agent.py)

```python
from .providers import get_llm_model
from .dependencies import HybridRAGDependencies

agent = Agent(
    get_llm_model(),
    deps_type=HybridRAGDependencies,
    system_prompt=SYSTEM_PROMPT
)
```

### With CLI (cli.py)

```python
from .settings import settings
from .dependencies import HybridRAGDependencies, SearchPreferences

embedding_client = openai.AsyncOpenAI(
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key
)
deps = HybridRAGDependencies(
    embedding_client=embedding_client,
    embedding_model=settings.embedding_model
)
```

### With Tools (in agent.py)

```python
@agent.tool
async def vector_search(ctx: RunContext[HybridRAGDependencies], query: str) -> List[Dict]:
    # Access embedding client via ctx.deps
    response = await ctx.deps.embedding_client.embeddings.create(
        input=query,
        model=ctx.deps.embedding_model
    )
    embedding = response.data[0].embedding
    # ... use embedding for search
```

---

## 10. Validation Checklist

Before implementation, verify:

- [ ] All required environment variables defined
- [ ] Settings class has proper field types and defaults
- [ ] Providers module has get_llm_model() and get_embedding_client()
- [ ] Dependencies dataclass includes embedding_client and embedding_model
- [ ] pyproject.toml includes all required packages
- [ ] .env.example documents all variables with examples
- [ ] Test fixtures mock embedding client properly
