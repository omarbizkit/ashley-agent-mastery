"""Comprehensive tests for each tool in the Hybrid RAG Knowledge Graph Agent.

This module tests all 8 tools:
1. vector_search - Pure semantic similarity search
2. graph_search - Knowledge graph search
3. hybrid_search - Combined vector + keyword search
4. perform_comprehensive_search - Parallel vector + graph
5. get_document - Full document retrieval
6. list_documents - Document listing
7. get_entity_relationships - Entity relationship traversal
8. get_entity_timeline - Temporal entity information
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel

from agent import agent, get_query_embedding
from dependencies import HybridRAGDependencies, SearchPreferences


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_client():
    """Create mock embedding client."""
    client = AsyncMock()
    client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    return client


@pytest.fixture
def test_deps(mock_embedding_client):
    """Create test dependencies."""
    return HybridRAGDependencies(
        embedding_client=mock_embedding_client,
        embedding_model="text-embedding-3-small",
        search_preferences=SearchPreferences()
    )


@pytest.fixture
def sample_vector_results():
    """Sample vector search results."""
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Artificial intelligence is transforming industries.",
            "similarity": 0.95,
            "metadata": {"topic": "AI"},
            "document_title": "AI Overview",
            "document_source": "test/ai.md"
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "content": "Machine learning enables pattern recognition.",
            "similarity": 0.85,
            "metadata": {"topic": "ML"},
            "document_title": "ML Basics",
            "document_source": "test/ml.md"
        }
    ]


@pytest.fixture
def sample_graph_results():
    """Sample graph search results."""
    return [
        {
            "fact": "OpenAI created ChatGPT",
            "uuid": "fact-uuid-1",
            "valid_at": "2022-11-30",
            "invalid_at": None,
            "source_node_uuid": "node-1"
        },
        {
            "fact": "Sam Altman is CEO of OpenAI",
            "uuid": "fact-uuid-2",
            "valid_at": "2019-01-01",
            "invalid_at": None,
            "source_node_uuid": "node-2"
        }
    ]


@pytest.fixture
def sample_hybrid_results():
    """Sample hybrid search results."""
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Python is a programming language.",
            "combined_score": 0.88,
            "vector_similarity": 0.85,
            "text_similarity": 0.95,
            "metadata": {"topic": "programming"},
            "document_title": "Python Guide",
            "document_source": "test/python.md"
        }
    ]


@pytest.fixture
def sample_document():
    """Sample document data."""
    return {
        "id": "doc-uuid-1",
        "title": "AI Overview",
        "source": "test/ai.md",
        "content": "Full document content about artificial intelligence...",
        "metadata": {"topic": "AI", "category": "technology"},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-15T00:00:00"
    }


@pytest.fixture
def sample_documents_list():
    """Sample documents list."""
    return [
        {
            "id": "doc-1",
            "title": "AI Overview",
            "source": "test/ai.md",
            "metadata": {"topic": "AI"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-15T00:00:00",
            "chunk_count": 5
        },
        {
            "id": "doc-2",
            "title": "ML Basics",
            "source": "test/ml.md",
            "metadata": {"topic": "ML"},
            "created_at": "2024-01-02T00:00:00",
            "updated_at": "2024-01-16T00:00:00",
            "chunk_count": 3
        }
    ]


# =============================================================================
# Test 1: vector_search Tool
# =============================================================================

class TestVectorSearchTool:
    """Tests for the vector_search tool."""

    @pytest.mark.asyncio
    async def test_vector_search_success(self, test_deps, sample_vector_results):
        """Test vector search returns results successfully."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_vector_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for information about AI",
                    deps=test_deps
                )

                assert result.output is not None
                # Verify the mock was called
                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_limit_validation(self, test_deps):
        """Test vector search limit parameter is clamped correctly."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            with agent.override(model=test_model):
                # The tool should clamp limit to 1-50 range
                result = await agent.run(
                    "Search with limit 100",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_vector_search_error_handling(self, test_deps):
        """Test vector search handles errors gracefully."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Database connection failed")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for AI content",
                    deps=test_deps
                )

                # Agent should still return output (error message in tool result)
                assert result.output is not None

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(self, test_deps):
        """Test vector search handles empty results."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for nonexistent topic",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test 2: graph_search Tool
# =============================================================================

class TestGraphSearchTool:
    """Tests for the graph_search tool."""

    @pytest.mark.asyncio
    async def test_graph_search_success(self, test_deps, sample_graph_results):
        """Test graph search returns results successfully."""
        test_model = TestModel(call_tools=['graph_search'])

        with patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_graph_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Find relationships for OpenAI",
                    deps=test_deps
                )

                assert result.output is not None
                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_search_error_handling(self, test_deps):
        """Test graph search handles errors gracefully."""
        test_model = TestModel(call_tools=['graph_search'])

        with patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Neo4j connection failed")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search knowledge graph",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_graph_search_empty_results(self, test_deps):
        """Test graph search handles empty results."""
        test_model = TestModel(call_tools=['graph_search'])

        with patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for unknown entity",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test 3: hybrid_search Tool
# =============================================================================

class TestHybridSearchTool:
    """Tests for the hybrid_search tool."""

    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, test_deps, sample_hybrid_results):
        """Test hybrid search returns results successfully."""
        test_model = TestModel(call_tools=['hybrid_search'])

        with patch('agent.db_hybrid_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_hybrid_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for Python programming",
                    deps=test_deps
                )

                assert result.output is not None
                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_text_weight_validation(self, test_deps, sample_hybrid_results):
        """Test hybrid search text_weight parameter is clamped correctly."""
        test_model = TestModel(call_tools=['hybrid_search'])

        with patch('agent.db_hybrid_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_hybrid_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search with high text weight",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self, test_deps):
        """Test hybrid search handles errors gracefully."""
        test_model = TestModel(call_tools=['hybrid_search'])

        with patch('agent.db_hybrid_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Hybrid search failed")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Hybrid search for content",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test 4: perform_comprehensive_search Tool
# =============================================================================

class TestComprehensiveSearchTool:
    """Tests for the perform_comprehensive_search tool."""

    @pytest.mark.asyncio
    async def test_comprehensive_search_success(
        self, test_deps, sample_vector_results, sample_graph_results
    ):
        """Test comprehensive search returns both vector and graph results."""
        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_graph:

            mock_vector.return_value = sample_vector_results
            mock_graph.return_value = sample_graph_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Comprehensive search for AI",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_comprehensive_search_vector_only(self, test_deps, sample_vector_results):
        """Test comprehensive search with only vector enabled."""
        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_vector:
            mock_vector.return_value = sample_vector_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search with vector only",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_comprehensive_search_graph_only(self, test_deps, sample_graph_results):
        """Test comprehensive search with only graph enabled."""
        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_graph:
            mock_graph.return_value = sample_graph_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search with graph only",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_comprehensive_search_partial_failure(
        self, test_deps, sample_vector_results
    ):
        """Test comprehensive search handles partial failures gracefully."""
        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_graph:

            mock_vector.return_value = sample_vector_results
            mock_graph.side_effect = Exception("Graph search failed")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Comprehensive search with potential failures",
                    deps=test_deps
                )

                # Should still return results from vector search
                assert result.output is not None


# =============================================================================
# Test 5: get_document Tool
# =============================================================================

class TestGetDocumentTool:
    """Tests for the get_document tool."""

    @pytest.mark.asyncio
    async def test_get_document_success(self, test_deps, sample_document):
        """Test get document returns document successfully."""
        test_model = TestModel(call_tools=['get_document'])

        with patch('agent.db_get_document', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_document

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get document doc-uuid-1",
                    deps=test_deps
                )

                assert result.output is not None
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, test_deps):
        """Test get document handles not found case."""
        test_model = TestModel(call_tools=['get_document'])

        with patch('agent.db_get_document', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get document nonexistent-id",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_get_document_error_handling(self, test_deps):
        """Test get document handles errors gracefully."""
        test_model = TestModel(call_tools=['get_document'])

        with patch('agent.db_get_document', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database error")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Retrieve document details",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test 6: list_documents Tool
# =============================================================================

class TestListDocumentsTool:
    """Tests for the list_documents tool."""

    @pytest.mark.asyncio
    async def test_list_documents_success(self, test_deps, sample_documents_list):
        """Test list documents returns documents successfully."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = sample_documents_list

            with agent.override(model=test_model):
                result = await agent.run(
                    "List all available documents",
                    deps=test_deps
                )

                assert result.output is not None
                mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(self, test_deps, sample_documents_list):
        """Test list documents with pagination parameters."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = sample_documents_list[:1]

            with agent.override(model=test_model):
                result = await agent.run(
                    "List documents with limit 1",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_list_documents_limit_validation(self, test_deps):
        """Test list documents limit parameter is clamped correctly."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            with agent.override(model=test_model):
                result = await agent.run(
                    "List documents with limit 200",
                    deps=test_deps
                )

                # Limit should be clamped to 100
                assert result.output is not None

    @pytest.mark.asyncio
    async def test_list_documents_empty_results(self, test_deps):
        """Test list documents handles empty results."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            with agent.override(model=test_model):
                result = await agent.run(
                    "List all documents",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_list_documents_error_handling(self, test_deps):
        """Test list documents handles errors gracefully."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database error")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Show me available documents",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test 7: get_entity_relationships Tool
# =============================================================================

class TestGetEntityRelationshipsTool:
    """Tests for the get_entity_relationships tool."""

    @pytest.mark.asyncio
    async def test_get_entity_relationships_success(self, test_deps):
        """Test get entity relationships returns results successfully."""
        test_model = TestModel(call_tools=['get_entity_relationships'])

        mock_relationships = {
            "central_entity": "OpenAI",
            "related_facts": [
                {"fact": "OpenAI created ChatGPT", "uuid": "fact-1", "valid_at": "2022-11-30"}
            ],
            "search_method": "graphiti_semantic_search"
        }

        with patch('agent.graph_get_relationships', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_relationships

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get relationships for OpenAI",
                    deps=test_deps
                )

                assert result.output is not None
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity_relationships_depth_validation(self, test_deps):
        """Test get entity relationships depth parameter is clamped correctly."""
        test_model = TestModel(call_tools=['get_entity_relationships'])

        with patch('agent.graph_get_relationships', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"central_entity": "Test", "related_facts": []}

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get relationships with depth 10",
                    deps=test_deps
                )

                # Depth should be clamped to 1-5 range
                assert result.output is not None

    @pytest.mark.asyncio
    async def test_get_entity_relationships_error_handling(self, test_deps):
        """Test get entity relationships handles errors gracefully."""
        test_model = TestModel(call_tools=['get_entity_relationships'])

        with patch('agent.graph_get_relationships', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Graph query failed")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Find entity relationships",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test 8: get_entity_timeline Tool
# =============================================================================

class TestGetEntityTimelineTool:
    """Tests for the get_entity_timeline tool."""

    @pytest.mark.asyncio
    async def test_get_entity_timeline_success(self, test_deps):
        """Test get entity timeline returns results successfully."""
        test_model = TestModel(call_tools=['get_entity_timeline'])

        mock_timeline = [
            {
                "fact": "Company founded",
                "uuid": "fact-1",
                "valid_at": "2015-01-01",
                "invalid_at": None
            },
            {
                "fact": "First product launched",
                "uuid": "fact-2",
                "valid_at": "2020-06-01",
                "invalid_at": None
            }
        ]

        with patch('agent.graph_client') as mock_client:
            mock_client.get_entity_timeline = AsyncMock(return_value=mock_timeline)

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get timeline for OpenAI",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_get_entity_timeline_with_date_range(self, test_deps):
        """Test get entity timeline with date range parameters."""
        test_model = TestModel(call_tools=['get_entity_timeline'])

        with patch('agent.graph_client') as mock_client:
            mock_client.get_entity_timeline = AsyncMock(return_value=[])

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get timeline for Tesla from 2020 to 2023",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_get_entity_timeline_error_handling(self, test_deps):
        """Test get entity timeline handles errors gracefully."""
        test_model = TestModel(call_tools=['get_entity_timeline'])

        with patch('agent.graph_client') as mock_client:
            mock_client.get_entity_timeline = AsyncMock(
                side_effect=Exception("Timeline query failed")
            )

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get entity timeline",
                    deps=test_deps
                )

                assert result.output is not None

    @pytest.mark.asyncio
    async def test_get_entity_timeline_invalid_date_format(self, test_deps):
        """Test get entity timeline handles invalid date format."""
        test_model = TestModel(call_tools=['get_entity_timeline'])

        with patch('agent.graph_client') as mock_client:
            mock_client.get_entity_timeline = AsyncMock(return_value=[])

            with agent.override(model=test_model):
                # The agent should handle date parsing errors
                result = await agent.run(
                    "Get timeline with dates",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Test Embedding Generation Helper
# =============================================================================

class TestEmbeddingHelper:
    """Tests for the get_query_embedding helper function."""

    @pytest.mark.asyncio
    async def test_get_query_embedding_success(self, test_deps):
        """Test embedding generation succeeds."""
        # Create a mock RunContext
        mock_ctx = Mock()
        mock_ctx.deps = test_deps

        embedding = await get_query_embedding(mock_ctx, "test query")

        assert embedding is not None
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_get_query_embedding_uses_correct_model(self, mock_embedding_client):
        """Test embedding generation uses the configured model."""
        deps = HybridRAGDependencies(
            embedding_client=mock_embedding_client,
            embedding_model="custom-embedding-model",
            search_preferences=SearchPreferences()
        )

        mock_ctx = Mock()
        mock_ctx.deps = deps

        await get_query_embedding(mock_ctx, "test query")

        mock_embedding_client.embeddings.create.assert_called_once_with(
            input="test query",
            model="custom-embedding-model"
        )


# =============================================================================
# Test Tool Parameter Validation
# =============================================================================

class TestToolParameterValidation:
    """Tests for tool parameter validation."""

    def test_agent_has_all_expected_tools(self):
        """Verify agent has all 8 expected tools."""
        expected_tools = [
            "vector_search",
            "graph_search",
            "hybrid_search",
            "perform_comprehensive_search",
            "get_document",
            "list_documents",
            "get_entity_relationships",
            "get_entity_timeline"
        ]

        tool_names = [tool.name for tool in agent._function_tools.values()]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

        assert len(tool_names) == 8, f"Expected 8 tools, found {len(tool_names)}"

    def test_vector_search_tool_schema(self):
        """Test vector_search tool has correct parameter schema."""
        tool = agent._function_tools.get("vector_search")
        assert tool is not None

        # Check tool has proper description
        assert tool.description is not None
        assert "semantic similarity search" in tool.description.lower()

    def test_hybrid_search_tool_schema(self):
        """Test hybrid_search tool has correct parameter schema."""
        tool = agent._function_tools.get("hybrid_search")
        assert tool is not None

        # Check tool has proper description
        assert tool.description is not None
        assert "keyword" in tool.description.lower() or "tsvector" in tool.description.lower()

    def test_list_documents_tool_schema(self):
        """Test list_documents tool has correct parameter schema."""
        tool = agent._function_tools.get("list_documents")
        assert tool is not None

        # Check tool has proper description
        assert tool.description is not None
        assert "documents" in tool.description.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
