"""Hybrid RAG Knowledge Graph Agent with multi-source search capabilities."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from pydantic_ai import Agent, RunContext

from providers import get_llm_model
from dependencies import HybridRAGDependencies
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


SYSTEM_PROMPT = """You are an intelligent AI assistant with access to multiple search systems for comprehensive information retrieval.

**Your Search Tools:**
- **vector_search**: Pure semantic similarity search across document chunks
- **hybrid_search**: Combines semantic + keyword matching (use when exact terms matter)
- **graph_search**: Finds entity relationships and facts in the knowledge graph
- **perform_comprehensive_search**: Runs vector + graph searches in parallel for thorough answers
- **get_document / list_documents**: Retrieve full documents when detailed context is needed
- **get_entity_relationships / get_entity_timeline**: Explore entity connections and temporal information

**When to Use Each:**
- General questions: Use vector_search or hybrid_search
- Questions about relationships between entities: Use graph_search
- Complex questions needing multiple sources: Use perform_comprehensive_search
- Questions with specific keywords or technical terms: Use hybrid_search with higher text_weight
- Time-sensitive questions: Use get_entity_timeline

**Response Guidelines:**
- Always search before answering factual questions
- Cite sources by mentioning document titles or entity names
- Be transparent when information is incomplete or uncertain
- Structure responses clearly with relevant details first
"""


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
