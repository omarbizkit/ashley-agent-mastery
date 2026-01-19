# Hybrid RAG Knowledge Graph Agent - Tool Specifications

This document specifies the 8 tools for the Hybrid RAG Knowledge Graph Agent. Each tool is MINIMAL and FOCUSED, doing one thing well.

---

## Tool Overview

| # | Tool Name | Purpose | Utility Function |
|---|-----------|---------|------------------|
| 1 | `vector_search` | Pure semantic similarity search | `db_utils.vector_search()` |
| 2 | `graph_search` | Knowledge graph search in Neo4j | `graph_utils.search_knowledge_graph()` |
| 3 | `hybrid_search` | Combined vector + TSVector keyword search | `db_utils.hybrid_search()` |
| 4 | `perform_comprehensive_search` | Parallel vector + graph search | Multiple utilities |
| 5 | `get_document` | Full document retrieval by ID | `db_utils.get_document()` |
| 6 | `list_documents` | Document listing with pagination | `db_utils.list_documents()` |
| 7 | `get_entity_relationships` | Entity relationship traversal | `graph_utils.get_entity_relationships()` |
| 8 | `get_entity_timeline` | Temporal entity information | `graph_utils.graph_client.get_entity_timeline()` |

---

## Tool 1: vector_search

**Purpose**: Performs pure semantic similarity search across document chunks using pgvector embeddings.

**Decorator**: `@agent.tool` (requires context for embedding client)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query string |
| `limit` | `int` | `10` | Maximum results to return (1-50) |

### Return Type

`List[Dict[str, Any]]` - List of chunks with:
- `chunk_id`: Chunk UUID
- `document_id`: Parent document UUID
- `content`: Chunk text content
- `similarity`: Cosine similarity score (0-1)
- `metadata`: Chunk metadata dict
- `document_title`: Parent document title
- `document_source`: Parent document source

### Wraps

```python
from utils.db_utils import vector_search as db_vector_search

# Internal call pattern:
embedding = await get_query_embedding(ctx, query)  # Generate embedding from query
results = await db_vector_search(embedding, limit)
```

### Error Handling

- Clamp `limit` to range [1, 50]
- Catch all exceptions and return `[{"error": str(e)}]`
- Log errors via `logger.error()`

---

## Tool 2: graph_search

**Purpose**: Searches the knowledge graph in Neo4j for entities, relationships, and temporal facts via Graphiti.

**Decorator**: `@agent.tool` (requires context)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query for finding related entities and facts |

### Return Type

`List[Dict[str, Any]]` - List of structured facts with:
- `fact`: The fact text
- `uuid`: Unique identifier
- `valid_at`: When fact became valid (ISO string or None)
- `invalid_at`: When fact was invalidated (ISO string or None)
- `source_node_uuid`: Reference to source node (UUID string or None)

### Wraps

```python
from utils.graph_utils import search_knowledge_graph

# Internal call pattern:
results = await search_knowledge_graph(query)
```

### Error Handling

- Catch all exceptions and return `[{"error": str(e)}]`
- Log errors via `logger.error()`

---

## Tool 3: hybrid_search

**Purpose**: Performs combined semantic vector search AND keyword search using PostgreSQL's TSVector for situations requiring both semantic understanding and specific keyword matches.

**Decorator**: `@agent.tool` (requires context for embedding client)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query string |
| `limit` | `int` | `10` | Maximum results to return (1-50) |
| `text_weight` | `float` | `0.3` | Weight for text similarity vs semantic (0.0-1.0) |

### Return Type

`List[Dict[str, Any]]` - List of chunks ranked by combined score with:
- `chunk_id`: Chunk UUID
- `document_id`: Parent document UUID
- `content`: Chunk text content
- `combined_score`: Combined ranking score
- `vector_similarity`: Vector similarity score
- `text_similarity`: TSVector text similarity score
- `metadata`: Chunk metadata dict
- `document_title`: Parent document title
- `document_source`: Parent document source

### Wraps

```python
from utils.db_utils import hybrid_search as db_hybrid_search

# Internal call pattern:
embedding = await get_query_embedding(ctx, query)
results = await db_hybrid_search(embedding, query, limit, text_weight)
```

### Error Handling

- Clamp `limit` to range [1, 50]
- Clamp `text_weight` to range [0.0, 1.0]
- Catch all exceptions and return `[{"error": str(e)}]`
- Log errors via `logger.error()`

---

## Tool 4: perform_comprehensive_search

**Purpose**: The master search function that combines vector search results with knowledge graph results in PARALLEL for comprehensive analysis.

**Decorator**: `@agent.tool` (requires context for embedding client)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query string |
| `use_vector` | `bool` | `True` | Whether to include vector search results |
| `use_graph` | `bool` | `True` | Whether to include graph search results |
| `limit` | `int` | `10` | Maximum number of vector results (1-50) |

### Return Type

`Dict[str, Any]` - Dictionary with:
- `vector_results`: List of vector search results (or error dict)
- `graph_results`: List of graph search results (or error dict)

### Wraps

```python
from utils.db_utils import vector_search as db_vector_search
from utils.graph_utils import search_knowledge_graph

# Internal call pattern (parallel execution):
tasks = []
if use_vector:
    embedding = await get_query_embedding(ctx, query)
    tasks.append(db_vector_search(embedding, limit))
if use_graph:
    tasks.append(search_knowledge_graph(query))

results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Error Handling

- Clamp `limit` to range [1, 50]
- Use `asyncio.gather(*tasks, return_exceptions=True)` to handle partial failures
- If a task returns an Exception, wrap it as `[{"error": str(exception)}]`
- Log errors via `logger.error()`
- Return `{"error": str(e)}` for complete failures

---

## Tool 5: get_document

**Purpose**: Retrieves complete document content with metadata when full context is needed.

**Decorator**: `@agent.tool` (requires context)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `document_id` | `str` | Required | UUID of the document to retrieve |

### Return Type

`Dict[str, Any]` - Complete document with:
- `id`: Document UUID
- `title`: Document title
- `source`: Document source path
- `content`: Full document content
- `metadata`: Document metadata dict
- `created_at`: Creation timestamp (ISO string)
- `updated_at`: Last update timestamp (ISO string)

Or error dict: `{"error": "Document not found: {document_id}"}`

### Wraps

```python
from utils.db_utils import get_document as db_get_document

# Internal call pattern:
result = await db_get_document(document_id)
```

### Error Handling

- Return `{"error": "Document not found: {document_id}"}` if not found
- Catch all exceptions and return `{"error": str(e)}`
- Log errors via `logger.error()`

---

## Tool 6: list_documents

**Purpose**: Lists available documents with metadata for browsing and discovery.

**Decorator**: `@agent.tool` (requires context)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `int` | `20` | Maximum documents to return (1-100) |
| `offset` | `int` | `0` | Number of documents to skip for pagination |

### Return Type

`List[Dict[str, Any]]` - List of documents with:
- `id`: Document UUID
- `title`: Document title
- `source`: Document source path
- `metadata`: Document metadata dict
- `created_at`: Creation timestamp (ISO string)
- `updated_at`: Last update timestamp (ISO string)
- `chunk_count`: Number of chunks for this document

### Wraps

```python
from utils.db_utils import list_documents as db_list_documents

# Internal call pattern:
results = await db_list_documents(limit, offset)
```

### Error Handling

- Clamp `limit` to range [1, 100]
- Clamp `offset` to minimum 0
- Catch all exceptions and return `[{"error": str(e)}]`
- Log errors via `logger.error()`

---

## Tool 7: get_entity_relationships

**Purpose**: Traverses the knowledge graph to find relationships for a specific entity.

**Decorator**: `@agent.tool` (requires context)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_name` | `str` | Required | Name of the entity to explore |
| `depth` | `int` | `2` | Maximum traversal depth (1-5) |

### Return Type

`Dict[str, Any]` - Entity relationships including:
- `central_entity`: The queried entity name
- `related_facts`: List of fact dicts with `fact`, `uuid`, `valid_at`
- `search_method`: Method used (e.g., "graphiti_semantic_search")

### Wraps

```python
from utils.graph_utils import get_entity_relationships as graph_get_relationships

# Internal call pattern:
results = await graph_get_relationships(entity_name, depth)
```

### Error Handling

- Clamp `depth` to range [1, 5]
- Catch all exceptions and return `{"error": str(e)}`
- Log errors via `logger.error()`

---

## Tool 8: get_entity_timeline

**Purpose**: Retrieves temporal information about an entity within a date range.

**Decorator**: `@agent.tool` (requires context)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_name` | `str` | Required | Name of the entity |
| `start_date` | `Optional[str]` | `None` | Start of date range (ISO format) |
| `end_date` | `Optional[str]` | `None` | End of date range (ISO format) |

### Return Type

`List[Dict[str, Any]]` - Timeline of facts with:
- `fact`: The fact text
- `uuid`: Unique identifier
- `valid_at`: When fact became valid (ISO string or None)
- `invalid_at`: When fact was invalidated (ISO string or None)

### Wraps

```python
from utils.graph_utils import graph_client

# Internal call pattern:
from datetime import datetime
start = datetime.fromisoformat(start_date) if start_date else None
end = datetime.fromisoformat(end_date) if end_date else None
results = await graph_client.get_entity_timeline(entity_name, start, end)
```

### Error Handling

- Parse ISO date strings with `datetime.fromisoformat()`
- Pass `None` for missing dates (no filtering)
- Catch all exceptions and return `[{"error": str(e)}]`
- Log errors via `logger.error()`

---

## Helper Function: get_query_embedding

**Purpose**: Generates embedding vector for a query string using the configured embedding client.

**Not a tool** - Internal helper function used by search tools.

### Implementation Pattern

```python
async def get_query_embedding(ctx: RunContext[HybridRAGDependencies], query: str) -> List[float]:
    """Generate embedding for a query string."""
    response = await ctx.deps.embedding_client.embeddings.create(
        input=query,
        model=ctx.deps.embedding_model
    )
    return response.data[0].embedding
```

---

## Dependencies Dataclass

The tools access dependencies through `ctx.deps` where deps is:

```python
@dataclass
class HybridRAGDependencies:
    """Dependencies for the Hybrid RAG Knowledge Graph Agent."""
    embedding_client: openai.AsyncOpenAI  # For generating query embeddings
    embedding_model: str = "text-embedding-3-small"  # Embedding model name
    search_preferences: SearchPreferences = field(default_factory=SearchPreferences)
    session_id: Optional[str] = None
```

---

## Common Error Handling Pattern

All tools follow this pattern:

```python
@agent.tool
async def tool_name(
    ctx: RunContext[HybridRAGDependencies],
    param: str
) -> ReturnType:
    """Tool docstring for LLM."""
    try:
        # Validate/clamp parameters
        # Call utility function
        # Log success
        return results
    except Exception as e:
        logger.error(f"Tool name failed: {e}")
        return {"error": f"Tool name failed: {str(e)}"}  # or list with error dict
```

---

## Implementation Notes

1. **Parameter Validation**: All numeric limits are clamped to valid ranges before use
2. **Logging**: All tools log their actions and errors for debugging
3. **Error Isolation**: Each tool catches its own exceptions and returns structured errors
4. **Parallel Execution**: `perform_comprehensive_search` uses `asyncio.gather()` for parallel search
5. **Context Access**: All tools access dependencies via `ctx.deps`
6. **No result_type**: Agent uses default string output, not structured output

---

## Testing Strategy

Each tool should be tested with:

1. **TestModel validation**: Use `pydantic_ai.models.test.TestModel` to verify tool registration
2. **Mock utilities**: Mock the underlying utility functions (`db_vector_search`, etc.)
3. **Parameter edge cases**: Test boundary values for limits, depths, weights
4. **Error handling**: Verify error messages are returned correctly
5. **Async patterns**: Test both sync (`run_sync`) and async (`run`) execution

Example test pattern:

```python
@pytest.mark.asyncio
async def test_vector_search_tool(test_dependencies):
    test_model = TestModel(call_tools=['vector_search'])

    with patch('agent.db_vector_search') as mock_search:
        mock_search.return_value = [
            {"content": "Test", "similarity": 0.9, "document_title": "Doc"}
        ]

        with agent.override(model=test_model):
            result = await agent.run("Search for test", deps=test_dependencies)
            assert "vector_search" in result.output
```
