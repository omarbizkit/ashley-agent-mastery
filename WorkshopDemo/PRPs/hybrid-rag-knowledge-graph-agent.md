---
name: "Hybrid RAG Knowledge Graph Agent PRP"
description: "Comprehensive PRP for building an intelligent AI assistant combining vector RAG with Neo4j knowledge graph capabilities"
confidence_score: 10
---

## Purpose

Build an intelligent AI assistant that combines traditional RAG with knowledge graph capabilities to provide comprehensive insights. The agent leverages:
- **Vector similarity search** via PostgreSQL with pgvector
- **Hybrid search** combining semantic and keyword matching with TSVector in PostgreSQL
- **Relationship-based reasoning** through Neo4j with Graphiti

This creates a powerful multi-layered search system that understands semantic content, keyword relevance, and entity relationships for superior information retrieval and analysis.

## Core Principles

1. **Pydantic AI Best Practices**: Deep integration with Pydantic AI patterns for agent creation, tools, and dependency injection
2. **Type Safety First**: Leverage Pydantic AI's type-safe design and Pydantic validation throughout
3. **Simple Yet Powerful**: Keep the agent minimal with focused tools that do one thing well
4. **Production Ready**: Include proper error handling, logging, and comprehensive testing

## Implementation Guidelines: Don't Over-Engineer

**IMPORTANT**: Keep your agent implementation focused and practical.

- **Start simple** - Build the minimum viable agent that meets requirements
- **Use existing utilities** - Copy over rag_pipeline utilities, don't rewrite them
- **Follow main_agent_reference** - Use proven patterns for settings, providers, CLI
- **Use string output by default** - No result_type unless structured output is specifically needed
- **Test early and often** - Use TestModel to validate as you build
- **Keep tools focused** - Each tool does one thing well

### Key Question:
**"Does this agent really need this feature to accomplish its core purpose?"**

If the answer is no, don't build it.

---

## Goal

Create a production-ready Pydantic AI agent with 8 focused tools that enable:
1. Pure semantic vector search across document chunks
2. Hybrid search combining semantic + keyword matching
3. Knowledge graph search for entity relationships
4. Comprehensive parallel search combining all capabilities
5. Document retrieval and listing
6. Entity relationship exploration
7. Temporal entity timeline queries

## Why

Traditional RAG systems only understand semantic similarity. By combining vector search with knowledge graphs, the agent can:
- Answer factual questions with high accuracy (vector search)
- Find exact keyword matches when needed (hybrid search)
- Understand relationships between entities (graph search)
- Provide temporal context for time-sensitive information
- Give comprehensive answers by combining multiple search approaches

## What

### Agent Type Classification
- [x] **Tool-Enabled Agent**: Agent with database and graph integration capabilities
- [x] **Chat Agent**: Conversational interface with streaming support

### External Integrations
- [x] PostgreSQL with pgvector (asyncpg connection pool)
- [x] Neo4j with Graphiti (knowledge graph)
- [x] OpenAI-compatible embeddings API

### Success Criteria
- [ ] All 8 tools work correctly with proper error handling
- [ ] Agent responds accurately using search results
- [ ] CLI provides real-time streaming with tool call visibility
- [ ] Comprehensive test coverage with TestModel
- [ ] Environment-based configuration (no hardcoded values)

---

## All Needed Context

### Database Objects Naming Convention

**IMPORTANT**: All database objects use the `wd_` prefix (WorkshopDemo) to avoid conflicts with existing database objects.

| Object Type | Name | Description |
|-------------|------|-------------|
| Table | `wd_documents` | Stores source documents |
| Table | `wd_chunks` | Stores document chunks with embeddings |
| Function | `wd_match_chunks()` | Vector similarity search |
| Function | `wd_hybrid_search()` | Combined vector + keyword search |
| Function | `wd_get_document_chunks()` | Get chunks for a document |
| Function | `wd_update_updated_at_column()` | Trigger function for timestamps |
| Trigger | `wd_update_documents_updated_at` | Auto-update timestamps |

### Project Structure (Copy from rag_pipeline)

```
PRPs/examples/rag_pipeline/
├── sql/
│   └── schema.sql          # DB schema with wd_match_chunks() and wd_hybrid_search() functions
├── utils/
│   ├── db_utils.py         # DatabasePool, vector_search(), hybrid_search(), get_document()
│   ├── graph_utils.py      # GraphitiClient, search_knowledge_graph(), get_entity_relationships()
│   ├── models.py           # Pydantic models for search results
│   └── providers.py        # get_llm_model(), get_embedding_client()
└── ingestion/              # Ingestion pipeline (already set up)
```

### Critical Files to Reference

**1. PRPs/examples/rag_pipeline/sql/schema.sql** - SQL functions for search:
```sql
-- Vector search function (wd_ prefix to avoid conflicts)
CREATE OR REPLACE FUNCTION wd_match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
) RETURNS TABLE (...)

-- Hybrid search function (vector + TSVector)
CREATE OR REPLACE FUNCTION wd_hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
) RETURNS TABLE (...)
```

**2. PRPs/examples/rag_pipeline/utils/db_utils.py** - Database utilities:
```python
# Key functions to use:
async def vector_search(embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]
async def hybrid_search(embedding: List[float], query_text: str, limit: int = 10, text_weight: float = 0.3) -> List[Dict[str, Any]]
async def get_document(document_id: str) -> Optional[Dict[str, Any]]
async def list_documents(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]
```

**3. PRPs/examples/rag_pipeline/utils/graph_utils.py** - Graphiti utilities:
```python
# Key functions to use:
async def search_knowledge_graph(query: str) -> List[Dict[str, Any]]
async def get_entity_relationships(entity: str, depth: int = 2) -> Dict[str, Any]
async def get_entity_timeline(entity_name: str, start_date: Optional[datetime], end_date: Optional[datetime]) -> List[Dict[str, Any]]
```

**4. PRPs/examples/main_agent_reference/settings.py** - Settings pattern:
```python
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    llm_provider: str = Field(default="openai")
    llm_api_key: str = Field(...)
    llm_model: str = Field(default="gpt-4")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
```

**5. PRPs/examples/main_agent_reference/providers.py** - Provider pattern:
```python
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return OpenAIModel(settings.llm_model, provider=provider)
```

### Pydantic AI Documentation References

IMPORTANT: Make sure you use the Context7 MCP Server for Pydantic AI and Gaphiti documentation to aid in your development.

**Agent Creation with Dependencies** (https://ai.pydantic.dev/dependencies):
```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent(
    'openai:gpt-5',
    deps_type=MyDeps,
)

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], query: str) -> str:
    # Access dependencies via ctx.deps
    response = await ctx.deps.http_client.get(...)
    return response.text
```

**Testing with TestModel** (https://ai.pydantic.dev/testing):
```python
from pydantic_ai.models.test import TestModel

def test_agent():
    test_model = TestModel()
    with agent.override(model=test_model):
        result = agent.run_sync("Test message", deps=deps)
        assert result.output is not None
```

### Graphiti Search Patterns (https://help.getzep.com/graphiti)

```python
# Hybrid search (semantic + BM25)
results = await graphiti.search(query)

# Results contain facts with temporal metadata
for result in results:
    print(result.fact)          # The fact text
    print(result.uuid)          # Unique identifier
    print(result.valid_at)      # When fact became valid
    print(result.invalid_at)    # When fact was invalidated (if applicable)
```

---

## Implementation Blueprint

### Prerequisites: Neo4j Setup

Before implementing the agent, ensure Neo4j is available:

**Option A: Local Neo4j (Docker)**
```bash
# Run Neo4j locally with Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest

# Verify it's running
curl http://localhost:7474
```

**Option B: Neo4j Aura (Cloud)**
1. Go to https://neo4j.com/cloud/aura/
2. Create a free AuraDB instance
3. Copy the connection URI (format: `neo4j+s://xxxxxxxx.databases.neo4j.io`)
4. Save the password

**Supabase Database Setup:**
1. Ensure pgvector extension is enabled in Supabase
2. Run the schema.sql file via Supabase SQL Editor
3. Get the connection string from: Dashboard > Project Settings > Database > Connection String

### Step 0: Copy RAG Pipeline Files

```bash
# Copy the entire rag_pipeline directory structure to the project root
cp -r PRPs/examples/rag_pipeline/* .
```

This provides:
- `sql/schema.sql` - Database schema and functions
- `utils/db_utils.py` - Database connection and search utilities
- `utils/graph_utils.py` - Graphiti knowledge graph utilities
- `utils/models.py` - Pydantic models for data validation
- `utils/providers.py` - LLM and embedding provider configuration

### Step 1: Create Agent Settings

**File: `settings.py`**

```python
"""Configuration management using pydantic-settings."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    llm_api_key: str = Field(..., description="API key for the LLM provider")
    llm_model: str = Field(default="gpt-4", description="Model name to use")
    llm_base_url: str = Field(default="https://api.openai.com/v1", description="Base URL for the LLM API")

    # Database Configuration
    database_url: str = Field(..., description="PostgreSQL connection URL")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    # Embedding Configuration
    embedding_api_key: str = Field(..., description="API key for embeddings")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    embedding_base_url: str = Field(default="https://api.openai.com/v1", description="Embedding API base URL")

    # Application Configuration
    log_level: str = Field(default="INFO")


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        raise ValueError(error_msg) from e


settings = load_settings()
```

### Step 2: Create Provider Configuration

**File: `providers.py`**

```python
"""Flexible provider configuration for LLM and embedding models."""

from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from .settings import settings


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """Get LLM model configuration based on environment variables."""
    llm_choice = model_choice or settings.llm_model
    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """Get embedding client for generating vectors."""
    return openai.AsyncOpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key
    )
```

### Step 3: Create Agent Dependencies

**File: `dependencies.py`**

```python
"""Agent dependencies using dataclass pattern."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import openai


@dataclass
class SearchPreferences:
    """Configuration for search behavior."""
    use_vector: bool = True
    use_graph: bool = True
    default_limit: int = 10


@dataclass
class HybridRAGDependencies:
    """Dependencies for the Hybrid RAG Knowledge Graph Agent."""
    embedding_client: openai.AsyncOpenAI
    embedding_model: str = "text-embedding-3-small"
    search_preferences: SearchPreferences = field(default_factory=SearchPreferences)
    session_id: Optional[str] = None
```

### Step 4: Create Agent with Tools

**File: `agent.py`**

```python
"""Hybrid RAG Knowledge Graph Agent with multi-source search capabilities."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from .providers import get_llm_model
from .dependencies import HybridRAGDependencies
from utils.db_utils import (
    vector_search as db_vector_search,
    hybrid_search as db_hybrid_search,
    get_document as db_get_document,
    list_documents as db_list_documents,
)
from utils.graph_utils import (
    search_knowledge_graph,
    get_entity_relationships as graph_get_relationships,
    graph_client,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an intelligent AI assistant with access to multiple search systems: a vector database for semantic search, TSVector for keyword matching, and a knowledge graph for entity relationships. Your primary capabilities include:

1. Vector search for pure semantic similarity
2. Hybrid search combining semantic and keyword matching using PostgreSQL's TSVector
3. Knowledge graph search for relationships and temporal facts in Neo4j
4. Comprehensive search that runs vector and graph searches in parallel
5. Full document retrieval when detailed context is needed

When answering questions, always search for relevant information before responding. Use hybrid_search when you need both semantic understanding and specific keyword matches. Use perform_comprehensive_search when you need to combine document chunks with entity relationships. Cite your sources by mentioning document titles and specific facts. Consider temporal aspects as some information may be time-sensitive. Look for relationships and connections between entities.

Your responses should be accurate and based on available data, well-structured and easy to understand, comprehensive while remaining concise, and transparent about information sources. Use the knowledge graph tool only when the user asks about relationships between two or more entities in the same question. Otherwise, use vector or hybrid search for document retrieval."""


# Create the agent - note: no result_type, defaults to string output
agent = Agent(
    get_llm_model(),
    deps_type=HybridRAGDependencies,
    system_prompt=SYSTEM_PROMPT
)


async def get_query_embedding(ctx: RunContext[HybridRAGDependencies], query: str) -> List[float]:
    """Generate embedding for a query string."""
    response = await ctx.deps.embedding_client.embeddings.create(
        input=query,
        model=ctx.deps.embedding_model
    )
    return response.data[0].embedding


@agent.tool
async def vector_search(
    ctx: RunContext[HybridRAGDependencies],
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Performs pure semantic similarity search across document chunks using pgvector embeddings.

    Args:
        query: Search query string
        limit: Maximum number of results to return (1-50)

    Returns:
        List of chunks with content, similarity scores, document metadata, and source information
    """
    try:
        limit = min(max(limit, 1), 50)
        embedding = await get_query_embedding(ctx, query)
        results = await db_vector_search(embedding, limit)
        logger.info(f"Vector search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return [{"error": f"Vector search failed: {str(e)}"}]


@agent.tool
async def graph_search(
    ctx: RunContext[HybridRAGDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Searches the knowledge graph in Neo4j for entities, relationships, and temporal facts.

    Args:
        query: Search query for finding related entities and facts

    Returns:
        List of structured facts with UUIDs, validity periods, and source node references
    """
    try:
        results = await search_knowledge_graph(query)
        logger.info(f"Graph search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return [{"error": f"Graph search failed: {str(e)}"}]


@agent.tool
async def hybrid_search(
    ctx: RunContext[HybridRAGDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Performs combined semantic vector search AND keyword search using PostgreSQL's TSVector.

    Args:
        query: Search query string
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs semantic (0.0-1.0, default 0.3)

    Returns:
        List of chunks ranked by combined score with both vector and text similarity scores
    """
    try:
        limit = min(max(limit, 1), 50)
        text_weight = min(max(text_weight, 0.0), 1.0)
        embedding = await get_query_embedding(ctx, query)
        results = await db_hybrid_search(embedding, query, limit, text_weight)
        logger.info(f"Hybrid search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return [{"error": f"Hybrid search failed: {str(e)}"}]


@agent.tool
async def perform_comprehensive_search(
    ctx: RunContext[HybridRAGDependencies],
    query: str,
    use_vector: bool = True,
    use_graph: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """
    The master search function that combines vector search results with knowledge graph results in parallel.

    Args:
        query: Search query string
        use_vector: Whether to include vector search results (default True)
        use_graph: Whether to include graph search results (default True)
        limit: Maximum number of vector results (1-50)

    Returns:
        Dictionary with both vector_results and graph_results for comprehensive analysis
    """
    try:
        limit = min(max(limit, 1), 50)
        tasks = []

        if use_vector:
            embedding = await get_query_embedding(ctx, query)
            tasks.append(db_vector_search(embedding, limit))

        if use_graph:
            tasks.append(search_knowledge_graph(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        response = {"vector_results": [], "graph_results": []}
        idx = 0

        if use_vector:
            if isinstance(results[idx], Exception):
                response["vector_results"] = [{"error": str(results[idx])}]
            else:
                response["vector_results"] = results[idx]
            idx += 1

        if use_graph:
            if isinstance(results[idx], Exception):
                response["graph_results"] = [{"error": str(results[idx])}]
            else:
                response["graph_results"] = results[idx]

        logger.info(f"Comprehensive search returned {len(response['vector_results'])} vector and {len(response['graph_results'])} graph results")
        return response
    except Exception as e:
        logger.error(f"Comprehensive search failed: {e}")
        return {"error": f"Comprehensive search failed: {str(e)}"}


@agent.tool
async def get_document(
    ctx: RunContext[HybridRAGDependencies],
    document_id: str
) -> Dict[str, Any]:
    """
    Retrieves complete document content with metadata when full context is needed.

    Args:
        document_id: UUID of the document to retrieve

    Returns:
        Complete document with content, title, source, metadata, and timestamps
    """
    try:
        result = await db_get_document(document_id)
        if result:
            logger.info(f"Retrieved document: {result.get('title', document_id)}")
            return result
        return {"error": f"Document not found: {document_id}"}
    except Exception as e:
        logger.error(f"Get document failed: {e}")
        return {"error": f"Failed to get document: {str(e)}"}


@agent.tool
async def list_documents(
    ctx: RunContext[HybridRAGDependencies],
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Lists available documents with metadata for browsing and discovery.

    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination

    Returns:
        List of documents with id, title, source, metadata, and chunk count
    """
    try:
        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)
        results = await db_list_documents(limit, offset)
        logger.info(f"Listed {len(results)} documents")
        return results
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return [{"error": f"Failed to list documents: {str(e)}"}]


@agent.tool
async def get_entity_relationships(
    ctx: RunContext[HybridRAGDependencies],
    entity_name: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Traverses the knowledge graph to find relationships for a specific entity.

    Args:
        entity_name: Name of the entity to explore
        depth: Maximum traversal depth (1-5)

    Returns:
        Entity relationships including related facts and search metadata
    """
    try:
        depth = min(max(depth, 1), 5)
        results = await graph_get_relationships(entity_name, depth)
        logger.info(f"Found relationships for entity: {entity_name}")
        return results
    except Exception as e:
        logger.error(f"Get entity relationships failed: {e}")
        return {"error": f"Failed to get entity relationships: {str(e)}"}


@agent.tool
async def get_entity_timeline(
    ctx: RunContext[HybridRAGDependencies],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves temporal information about an entity within a date range.

    Args:
        entity_name: Name of the entity
        start_date: Start of date range (ISO format, optional)
        end_date: End of date range (ISO format, optional)

    Returns:
        Timeline of facts about the entity with validity periods
    """
    try:
        from datetime import datetime
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        results = await graph_client.get_entity_timeline(entity_name, start, end)
        logger.info(f"Retrieved timeline for entity: {entity_name}")
        return results
    except Exception as e:
        logger.error(f"Get entity timeline failed: {e}")
        return [{"error": f"Failed to get entity timeline: {str(e)}"}]
```

### Step 5: Create CLI Interface

**File: `cli.py`**

```python
#!/usr/bin/env python3
"""Conversational CLI with real-time streaming and tool call visibility."""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import openai

from pydantic_ai import (
    Agent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FinalResultEvent,
)
from agent import agent
from dependencies import HybridRAGDependencies, SearchPreferences
from settings import settings
from utils.db_utils import initialize_database, close_database
from utils.graph_utils import initialize_graph, close_graph

console = Console()


async def stream_agent_interaction(user_input: str, deps: HybridRAGDependencies) -> str:
    """Stream agent interaction with real-time tool call display."""

    try:
        async with agent.iter(user_input, deps=deps) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    pass

                elif Agent.is_model_request_node(node):
                    console.print("[bold blue]Assistant:[/bold blue] ", end="")
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, TextPartDelta):
                                    console.print(event.delta.content_delta, end="")
                            elif isinstance(event, FinalResultEvent):
                                console.print()

                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as tool_stream:
                        async for event in tool_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                console.print(f"  [cyan]Calling:[/cyan] [bold]{event.part.tool_name}[/bold]")
                            elif isinstance(event, FunctionToolResultEvent):
                                result_preview = str(event.result.content)[:100]
                                console.print(f"  [green]Result:[/green] [dim]{result_preview}...[/dim]")

                elif Agent.is_end_node(node):
                    pass

        return run.result.output

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return f"Error: {e}"


async def main():
    """Main conversation loop."""

    welcome = Panel(
        "[bold blue]Hybrid RAG Knowledge Graph Agent[/bold blue]\n\n"
        "[green]Vector + Graph search with streaming[/green]\n"
        "[dim]Type 'exit' to quit[/dim]",
        style="blue",
        padding=(1, 2)
    )
    console.print(welcome)
    console.print()

    # Initialize connections
    console.print("[dim]Initializing database connections...[/dim]")
    await initialize_database()
    await initialize_graph()
    console.print("[green]Ready![/green]\n")

    # Create dependencies
    embedding_client = openai.AsyncOpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key
    )
    deps = HybridRAGDependencies(
        embedding_client=embedding_client,
        embedding_model=settings.embedding_model,
        search_preferences=SearchPreferences()
    )

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold green]You").strip()

                if user_input.lower() in ['exit', 'quit']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not user_input:
                    continue

                await stream_agent_interaction(user_input, deps)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
    finally:
        await close_database()
        await close_graph()
        await embedding_client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 6: Create pyproject.toml

**File: `pyproject.toml`**

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

### Step 7: Create .env.example

**File: `.env.example`**

```bash
# =============================================================================
# LLM Configuration (OpenAI-compatible)
# =============================================================================
LLM_API_KEY=sk-your-openai-api-key
LLM_CHOICE=gpt-4
LLM_BASE_URL=https://api.openai.com/v1

# =============================================================================
# Database Configuration (Supabase PostgreSQL)
# =============================================================================
# Format: postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
# Get this from: Supabase Dashboard > Project Settings > Database > Connection String > URI
DATABASE_URL=postgresql://postgres.xxxxxxxxxxxx:your-password@aws-0-us-east-1.pooler.supabase.com:6543/postgres

# =============================================================================
# Neo4j Configuration (for Knowledge Graph)
# =============================================================================
# Local: bolt://localhost:7687
# Aura Cloud: neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# =============================================================================
# Embedding Configuration
# =============================================================================
EMBEDDING_API_KEY=sk-your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
VECTOR_DIMENSION=1536
```

### Step 8: Create Package Init Files

**File: `utils/__init__.py`**

```python
"""Utilities package for database and graph operations."""

from .db_utils import (
    initialize_database,
    close_database,
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks,
)
from .graph_utils import (
    initialize_graph,
    close_graph,
    search_knowledge_graph,
    get_entity_relationships,
    graph_client,
)
from .providers import get_llm_model, get_embedding_client

__all__ = [
    "initialize_database",
    "close_database",
    "vector_search",
    "hybrid_search",
    "get_document",
    "list_documents",
    "get_document_chunks",
    "initialize_graph",
    "close_graph",
    "search_knowledge_graph",
    "get_entity_relationships",
    "graph_client",
    "get_llm_model",
    "get_embedding_client",
]
```

**File: `tests/__init__.py`**

```python
"""Test package."""
```

### Step 9: Create Tests

**File: `tests/test_agent.py`**

```python
"""Comprehensive tests for the Hybrid RAG Knowledge Graph Agent."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
from pydantic_ai.models.test import TestModel

from agent import agent
from dependencies import HybridRAGDependencies, SearchPreferences


@pytest.fixture
def mock_embedding_client():
    """Create mock embedding client."""
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


class TestAgentBasics:
    """Test basic agent functionality."""

    def test_agent_instantiation(self, test_dependencies):
        """Test agent can be instantiated and run."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = agent.run_sync(
                "What documents are available?",
                deps=test_dependencies
            )
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_async(self, test_dependencies):
        """Test async agent execution."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = await agent.run(
                "Search for information about AI",
                deps=test_dependencies
            )
            assert result.output is not None


class TestAgentTools:
    """Test agent tool functionality."""

    @pytest.mark.asyncio
    async def test_vector_search_tool(self, test_dependencies):
        """Test vector search tool."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search') as mock_search:
            mock_search.return_value = [
                {"content": "Test content", "similarity": 0.9, "document_title": "Test Doc"}
            ]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for test content",
                    deps=test_dependencies
                )
                assert "vector_search" in result.output

    @pytest.mark.asyncio
    async def test_graph_search_tool(self, test_dependencies):
        """Test graph search tool."""
        test_model = TestModel(call_tools=['graph_search'])

        with patch('agent.search_knowledge_graph') as mock_search:
            mock_search.return_value = [
                {"fact": "Entity A relates to Entity B", "uuid": "123"}
            ]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Find relationships between entities",
                    deps=test_dependencies
                )
                assert "graph_search" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Validation Loop

### Level 1: Project Structure Validation

```bash
# Verify all required files exist
test -f pyproject.toml && echo "✓ pyproject.toml"
test -f settings.py && echo "✓ settings.py"
test -f providers.py && echo "✓ providers.py"
test -f dependencies.py && echo "✓ dependencies.py"
test -f agent.py && echo "✓ agent.py"
test -f cli.py && echo "✓ cli.py"
test -f .env.example && echo "✓ .env.example"
test -d utils && echo "✓ utils/ directory"
test -f utils/__init__.py && echo "✓ utils/__init__.py"
test -f utils/db_utils.py && echo "✓ utils/db_utils.py"
test -f utils/graph_utils.py && echo "✓ utils/graph_utils.py"
test -d tests && echo "✓ tests/ directory"
test -f tests/__init__.py && echo "✓ tests/__init__.py"
test -f sql/schema.sql && echo "✓ sql/schema.sql"
```

### Level 2: Syntax and Import Validation

```bash
# Check Python syntax
python -m py_compile settings.py
python -m py_compile providers.py
python -m py_compile dependencies.py
python -m py_compile agent.py
python -m py_compile cli.py

# Verify imports work
python -c "from settings import settings; print('Settings loaded')"
python -c "from providers import get_llm_model; print('Providers loaded')"
python -c "from agent import agent; print(f'Agent has {len(agent.tools)} tools')"
```

### Level 3: Agent Functionality Validation

```bash
# Test agent with TestModel
python -c "
from pydantic_ai.models.test import TestModel
from unittest.mock import AsyncMock, Mock
from agent import agent
from dependencies import HybridRAGDependencies, SearchPreferences

# Create mock embedding client
mock_client = AsyncMock()
mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])

deps = HybridRAGDependencies(
    embedding_client=mock_client,
    embedding_model='test',
    search_preferences=SearchPreferences()
)

test_model = TestModel()
with agent.override(model=test_model):
    result = agent.run_sync('Test message', deps=deps)
    print(f'Agent response: {result.output[:100]}...')
    print('Agent validation passed!')
"
```

### Level 4: Comprehensive Test Suite

```bash
# Run pytest with coverage
uv run pytest tests/ -v --cov=. --cov-report=term-missing

# Run type checking
uv run mypy . --ignore-missing-imports

# Run linting
uv run ruff check --fix .
```

---

## Final Validation Checklist

### Agent Implementation Completeness

- [ ] Copy rag_pipeline files to project root
- [ ] Create `settings.py` with environment configuration
- [ ] Create `providers.py` with get_llm_model() function
- [ ] Create `dependencies.py` with HybridRAGDependencies dataclass
- [ ] Create `agent.py` with all 8 tools
- [ ] Create `cli.py` with streaming interface
- [ ] Create `.env.example` with all required variables
- [ ] Create `tests/test_agent.py` with comprehensive tests

### Tool Implementation

- [ ] `vector_search` - Pure semantic similarity search
- [ ] `graph_search` - Knowledge graph search
- [ ] `hybrid_search` - Combined vector + keyword search
- [ ] `perform_comprehensive_search` - Parallel vector + graph
- [ ] `get_document` - Full document retrieval
- [ ] `list_documents` - Document listing
- [ ] `get_entity_relationships` - Entity relationship traversal
- [ ] `get_entity_timeline` - Temporal entity information

### Best Practices

- [ ] No hardcoded model strings (use environment variables)
- [ ] No result_type (defaults to string output)
- [ ] Proper error handling in all tools
- [ ] Logging for debugging
- [ ] TestModel validation passes
- [ ] All imports work correctly

---

## Implementation Tasks (In Order)

1. **Set Up Prerequisites** - Neo4j (Docker or Aura) + Supabase pgvector
2. **Copy RAG Pipeline Files** - Copy `PRPs/examples/rag_pipeline/*` to project root maintaining structure
3. **Create pyproject.toml** - Dependencies and project configuration
4. **Create Settings** - `settings.py` with all environment variables
5. **Create Providers** - `providers.py` following main_agent_reference pattern
6. **Create Dependencies** - `dependencies.py` with dataclass for agent deps
7. **Create Agent** - `agent.py` with system prompt and all 8 tools
8. **Create CLI** - `cli.py` with Rich streaming interface
9. **Create Package Init Files** - `utils/__init__.py` and `tests/__init__.py`
10. **Create Environment Example** - `.env.example` with Supabase format
11. **Create Tests** - `tests/test_agent.py` with TestModel validation
12. **Run Schema SQL** - Execute `sql/schema.sql` in Supabase SQL Editor
13. **Validate All** - Run validation loop commands

---

## Anti-Patterns to Avoid

- **Don't skip copying rag_pipeline** - Use existing utilities, don't rewrite
- **Don't hardcode model strings** - Use `get_llm_model()` from providers
- **Don't add result_type** - Default string output is sufficient
- **Don't create complex dependencies** - Keep dataclass simple
- **Don't skip error handling** - Every tool should handle exceptions
- **Don't skip TestModel validation** - Test before using real LLM

---

## Confidence Score: 10/10

This PRP has maximum confidence for one-pass implementation because:

1. **All utilities exist** - rag_pipeline provides complete DB and graph utilities
2. **Proven patterns** - Following main_agent_reference exactly
3. **Clear structure** - Each file has a specific purpose
4. **Comprehensive context** - All code examples included inline
5. **Validation gates** - Executable commands to verify each step
6. **Complete dependencies** - pyproject.toml with all required packages
7. **Verified streaming API** - CLI uses correct Pydantic AI event types
8. **Package structure** - All `__init__.py` files included
9. **Database naming** - All objects use `wd_` prefix to avoid Supabase conflicts
10. **Infrastructure setup** - Neo4j and Supabase setup instructions included

**All previous risks addressed:**
- ✅ Database connection: Supabase URL format documented with example
- ✅ Neo4j setup: Docker and Aura Cloud options provided
- ✅ Import errors: Package `__init__.py` files included
- ✅ API mismatches: Streaming code verified against Pydantic AI docs
- ✅ Dependency issues: Complete pyproject.toml with versions
