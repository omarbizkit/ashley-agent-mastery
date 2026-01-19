# Hybrid RAG Knowledge Graph Agent

A production-ready Pydantic AI agent that combines vector search (pgvector), keyword search (TSVector), and knowledge graph traversal (Neo4j/Graphiti) for comprehensive information retrieval.

## Features

- **Multi-Source Search**: Combines semantic vector search, keyword matching, and knowledge graph traversal
- **8 Specialized Tools**: Purpose-built tools for different search patterns
- **Real-Time Streaming**: CLI with live tool call visibility and streaming responses
- **Production Ready**: Environment-based configuration, comprehensive error handling, and full test coverage

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid RAG Agent                         │
│                   (Pydantic AI + GPT-4)                     │
├─────────────────────────────────────────────────────────────┤
│  Tools:                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │vector_search│ │graph_search │ │hybrid_search        │   │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
│         │               │                    │              │
│  ┌──────┴──────┐ ┌──────┴──────┐ ┌──────────┴──────────┐   │
│  │get_document │ │list_docs    │ │comprehensive_search │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────────────┐ ┌─────────────────────┐           │
│  │get_entity_relations │ │get_entity_timeline  │           │
│  └─────────────────────┘ └─────────────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Data Sources:                                              │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ PostgreSQL      │  │ Neo4j           │                  │
│  │ (Supabase)      │  │ (Aura Free)     │                  │
│  │ - pgvector      │  │ - Graphiti      │                  │
│  │ - TSVector      │  │ - Knowledge     │                  │
│  │ - Documents     │  │   Graph         │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials:
# - LLM_API_KEY: OpenAI API key
# - DATABASE_URL: Supabase PostgreSQL connection string
# - NEO4J_URI, NEO4J_PASSWORD: Neo4j Aura credentials
```

### 3. Set Up Database

```bash
# Run the schema in your Supabase SQL editor
# Contents of sql/schema.sql
```

### 4. Run the Agent

```bash
# Interactive CLI mode
python cli.py

# Or use programmatically
python -c "
import asyncio
from agent import agent
from dependencies import HybridRAGDependencies
from providers import get_embedding_client
from settings import settings

async def main():
    deps = HybridRAGDependencies(
        embedding_client=get_embedding_client(),
        embedding_model=settings.embedding_model
    )
    result = await agent.run('What documents are available?', deps=deps)
    print(result.output)

asyncio.run(main())
"
```

## Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `vector_search` | Pure semantic similarity search via pgvector | General questions, concept matching |
| `graph_search` | Knowledge graph search in Neo4j | Entity relationships, facts |
| `hybrid_search` | Combined vector + TSVector keyword search | Technical terms, exact matches |
| `perform_comprehensive_search` | Parallel vector + graph search | Complex questions needing multiple sources |
| `get_document` | Full document retrieval by ID | Deep context on specific documents |
| `list_documents` | Document listing with pagination | Browsing available content |
| `get_entity_relationships` | Entity relationship traversal | Exploring connections |
| `get_entity_timeline` | Temporal entity information | Time-sensitive queries |

## Project Structure

```
hybrid-rag-agent/
├── agent.py              # Main agent with 8 tools
├── settings.py           # Environment configuration
├── providers.py          # LLM and embedding providers
├── dependencies.py       # Agent dependencies dataclass
├── cli.py                # Conversational CLI interface
├── pyproject.toml        # Package configuration
├── .env.example          # Environment template
├── utils/
│   ├── __init__.py
│   ├── db_utils.py       # PostgreSQL/pgvector operations
│   ├── graph_utils.py    # Neo4j/Graphiti operations
│   └── models.py         # Pydantic data models
├── sql/
│   └── schema.sql        # Database schema
├── tests/
│   ├── test_agent.py     # Agent tests
│   ├── test_tools.py     # Tool tests
│   ├── test_integration.py
│   ├── test_validation.py
│   ├── conftest.py       # Test fixtures
│   └── VALIDATION_REPORT.md
└── planning/
    ├── prompts.md        # System prompt specifications
    ├── tools.md          # Tool specifications
    └── dependencies.md   # Dependency specifications
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | - | OpenAI API key |
| `LLM_MODEL` | No | `gpt-4` | Model name |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | API base URL |
| `DATABASE_URL` | Yes | - | PostgreSQL connection URL |
| `EMBEDDING_API_KEY` | Yes | - | Embedding API key |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `NEO4J_URI` | Yes | - | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | - | Neo4j password |
| `USE_GRAPH_SEARCH` | No | `true` | Enable graph search |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test categories
python -m pytest tests/test_validation.py -v  # Validation gates
python -m pytest tests/test_tools.py -v       # Tool tests
python -m pytest tests/test_integration.py -v # Integration tests
```

## Usage Examples

### Basic Search

```python
from agent import agent
from dependencies import HybridRAGDependencies

# Vector search for semantic similarity
result = await agent.run(
    "Find information about machine learning algorithms",
    deps=deps
)
```

### Hybrid Search with Keywords

```python
# When exact terms matter
result = await agent.run(
    "Search for 'transformer architecture' with technical details",
    deps=deps
)
```

### Entity Exploration

```python
# Explore knowledge graph relationships
result = await agent.run(
    "What are the relationships between OpenAI and Microsoft?",
    deps=deps
)
```

### Comprehensive Multi-Source Search

```python
# Complex questions needing both vector and graph
result = await agent.run(
    "Give me a complete overview of Sam Altman's role in AI development",
    deps=deps
)
```

## CLI Features

The interactive CLI provides:

- **Real-time streaming**: See responses as they're generated
- **Tool visibility**: Watch which tools are being called
- **Conversation history**: Multi-turn conversations with context
- **Rich formatting**: Markdown rendering in terminal

```bash
python cli.py
```

Commands:
- Type your question and press Enter
- Type `exit` or `quit` to end the session
- Use Ctrl+C to interrupt a response

## Development

### Adding New Tools

1. Define the tool function in `agent.py`:

```python
@agent.tool
async def my_new_tool(
    ctx: RunContext[HybridRAGDependencies],
    param: str
) -> Dict[str, Any]:
    """Tool description for the LLM."""
    # Implementation
    return result
```

2. Update the system prompt to describe when to use the tool

3. Add tests in `tests/test_tools.py`

### Testing with TestModel

```python
from pydantic_ai.models.test import TestModel

def test_my_tool(test_dependencies):
    test_model = TestModel(call_tools=['my_new_tool'])

    with agent.override(model=test_model):
        result = agent.run_sync("Test prompt", deps=test_dependencies)
        assert result.output is not None
```

## License

MIT License

## Acknowledgments

- Built with [Pydantic AI](https://ai.pydantic.dev/)
- Vector search powered by [pgvector](https://github.com/pgvector/pgvector)
- Knowledge graph via [Graphiti](https://github.com/getzep/graphiti)
