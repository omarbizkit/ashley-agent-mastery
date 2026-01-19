"""Comprehensive tests for the Hybrid RAG Knowledge Graph Agent.

This module tests:
- Agent instantiation and configuration
- Basic agent functionality with TestModel
- Tool registration and availability
- Dependency injection patterns
- Sync and async execution
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart

from agent import agent, SYSTEM_PROMPT
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
def test_dependencies(mock_embedding_client):
    """Create test dependencies."""
    return HybridRAGDependencies(
        embedding_client=mock_embedding_client,
        embedding_model="text-embedding-3-small",
        search_preferences=SearchPreferences()
    )


# =============================================================================
# Agent Configuration Tests
# =============================================================================

class TestAgentConfiguration:
    """Test agent configuration and setup."""

    def test_agent_has_system_prompt(self):
        """Test agent has system prompt configured."""
        assert agent._system_prompts is not None
        assert len(agent._system_prompts) > 0

    def test_system_prompt_content(self):
        """Test system prompt contains key guidance."""
        assert "search" in SYSTEM_PROMPT.lower()
        assert "vector" in SYSTEM_PROMPT.lower() or "semantic" in SYSTEM_PROMPT.lower()

    def test_agent_deps_type(self):
        """Test agent has correct dependency type."""
        assert agent._deps_type == HybridRAGDependencies

    def test_agent_model_configured(self):
        """Test agent has a model configured."""
        # Agent should have a default model from providers.py
        assert agent.model is not None


# =============================================================================
# Agent Tool Tests
# =============================================================================

class TestAgentTools:
    """Test agent has all expected tools registered."""

    EXPECTED_TOOLS = [
        "vector_search",
        "graph_search",
        "hybrid_search",
        "perform_comprehensive_search",
        "get_document",
        "list_documents",
        "get_entity_relationships",
        "get_entity_timeline"
    ]

    def test_agent_has_all_tools(self):
        """Test agent has exactly 8 tools."""
        tool_names = list(agent._function_tools.keys())
        assert len(tool_names) == 8, f"Expected 8 tools, got {len(tool_names)}"

    def test_all_expected_tools_present(self):
        """Test all expected tools are registered."""
        tool_names = list(agent._function_tools.keys())
        for expected in self.EXPECTED_TOOLS:
            assert expected in tool_names, f"Missing tool: {expected}"

    def test_tools_have_descriptions(self):
        """Test all tools have descriptions."""
        for tool_name, tool in agent._function_tools.items():
            assert tool.description is not None, f"Tool {tool_name} has no description"
            assert len(tool.description) > 10, f"Tool {tool_name} has too short description"

    def test_vector_search_tool_exists(self):
        """Test vector_search tool is properly registered."""
        tool = agent._function_tools.get("vector_search")
        assert tool is not None
        assert "semantic" in tool.description.lower()

    def test_graph_search_tool_exists(self):
        """Test graph_search tool is properly registered."""
        tool = agent._function_tools.get("graph_search")
        assert tool is not None
        assert "knowledge graph" in tool.description.lower() or "neo4j" in tool.description.lower()

    def test_hybrid_search_tool_exists(self):
        """Test hybrid_search tool is properly registered."""
        tool = agent._function_tools.get("hybrid_search")
        assert tool is not None
        assert "keyword" in tool.description.lower() or "tsvector" in tool.description.lower()

    def test_comprehensive_search_tool_exists(self):
        """Test perform_comprehensive_search tool is properly registered."""
        tool = agent._function_tools.get("perform_comprehensive_search")
        assert tool is not None
        assert "parallel" in tool.description.lower() or "comprehensive" in tool.description.lower()

    def test_get_document_tool_exists(self):
        """Test get_document tool is properly registered."""
        tool = agent._function_tools.get("get_document")
        assert tool is not None
        assert "document" in tool.description.lower()

    def test_list_documents_tool_exists(self):
        """Test list_documents tool is properly registered."""
        tool = agent._function_tools.get("list_documents")
        assert tool is not None
        assert "documents" in tool.description.lower()

    def test_get_entity_relationships_tool_exists(self):
        """Test get_entity_relationships tool is properly registered."""
        tool = agent._function_tools.get("get_entity_relationships")
        assert tool is not None
        assert "relationship" in tool.description.lower()

    def test_get_entity_timeline_tool_exists(self):
        """Test get_entity_timeline tool is properly registered."""
        tool = agent._function_tools.get("get_entity_timeline")
        assert tool is not None
        assert "temporal" in tool.description.lower() or "timeline" in tool.description.lower()


# =============================================================================
# Agent Basic Functionality Tests
# =============================================================================

class TestAgentBasicFunctionality:
    """Test basic agent execution."""

    def test_agent_instantiation(self, test_dependencies):
        """Test agent can be instantiated and run."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = agent.run_sync(
                "What documents are available?",
                deps=test_dependencies
            )
            assert result.output is not None
            assert isinstance(result.output, str)

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
            assert isinstance(result.output, str)

    def test_agent_returns_string_output(self, test_dependencies):
        """Test agent returns string output (no result_type)."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = agent.run_sync("Hello", deps=test_dependencies)
            # Default is string output
            assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_agent_with_function_model(self, test_dependencies):
        """Test agent works with FunctionModel."""
        async def simple_response(messages, info: AgentInfo):
            return ModelResponse(
                parts=[TextPart(content="This is a test response.")]
            )

        function_model = FunctionModel(simple_response)

        with agent.override(model=function_model):
            result = await agent.run("Test prompt", deps=test_dependencies)
            assert result.output == "This is a test response."


# =============================================================================
# Agent Tool Execution Tests
# =============================================================================

class TestAgentToolExecution:
    """Test agent tool execution with mocking."""

    @pytest.mark.asyncio
    async def test_vector_search_tool_called(self, test_dependencies):
        """Test vector search tool is called correctly."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"content": "Test content", "similarity": 0.9, "document_title": "Test Doc"}
            ]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for test content",
                    deps=test_dependencies
                )
                assert result.output is not None
                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_search_tool_called(self, test_dependencies):
        """Test graph search tool is called correctly."""
        test_model = TestModel(call_tools=['graph_search'])

        with patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"fact": "Entity A relates to Entity B", "uuid": "123"}
            ]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Find relationships between entities",
                    deps=test_dependencies
                )
                assert result.output is not None
                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_tool_called(self, test_dependencies):
        """Test hybrid search tool is called correctly."""
        test_model = TestModel(call_tools=['hybrid_search'])

        with patch('agent.db_hybrid_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "content": "Test content",
                    "combined_score": 0.85,
                    "vector_similarity": 0.8,
                    "text_similarity": 0.9,
                    "document_title": "Test Doc"
                }
            ]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search with keywords",
                    deps=test_dependencies
                )
                assert result.output is not None
                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_tool_called(self, test_dependencies):
        """Test list documents tool is called correctly."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [
                {"id": "1", "title": "Doc 1", "source": "test.md", "chunk_count": 5}
            ]

            with agent.override(model=test_model):
                result = await agent.run(
                    "List all documents",
                    deps=test_dependencies
                )
                assert result.output is not None
                mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_tool_called(self, test_dependencies):
        """Test get document tool is called correctly."""
        test_model = TestModel(call_tools=['get_document'])

        with patch('agent.db_get_document', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "id": "doc-1",
                "title": "Test Document",
                "content": "Full content here...",
                "source": "test.md"
            }

            with agent.override(model=test_model):
                result = await agent.run(
                    "Get document doc-1",
                    deps=test_dependencies
                )
                assert result.output is not None
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_comprehensive_search_tool_called(self, test_dependencies):
        """Test comprehensive search tool is called correctly."""
        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_graph:

            mock_vector.return_value = [{"content": "Vector result", "similarity": 0.9}]
            mock_graph.return_value = [{"fact": "Graph result", "uuid": "123"}]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Comprehensive search for AI",
                    deps=test_dependencies
                )
                assert result.output is not None


# =============================================================================
# Dependency Tests
# =============================================================================

class TestDependencies:
    """Test dependency injection and configuration."""

    def test_search_preferences_defaults(self):
        """Test SearchPreferences has correct defaults."""
        prefs = SearchPreferences()
        assert prefs.use_vector is True
        assert prefs.use_graph is True
        assert prefs.default_limit == 10

    def test_search_preferences_custom(self):
        """Test SearchPreferences can be customized."""
        prefs = SearchPreferences(
            use_vector=False,
            use_graph=True,
            default_limit=20
        )
        assert prefs.use_vector is False
        assert prefs.use_graph is True
        assert prefs.default_limit == 20

    def test_hybrid_rag_dependencies_creation(self, mock_embedding_client):
        """Test HybridRAGDependencies creation."""
        deps = HybridRAGDependencies(
            embedding_client=mock_embedding_client,
            embedding_model="test-model"
        )
        assert deps.embedding_model == "test-model"
        assert deps.session_id is None

    def test_hybrid_rag_dependencies_with_session(self, mock_embedding_client):
        """Test HybridRAGDependencies with session ID."""
        deps = HybridRAGDependencies(
            embedding_client=mock_embedding_client,
            embedding_model="test-model",
            session_id="session-123"
        )
        assert deps.session_id == "session-123"

    def test_hybrid_rag_dependencies_default_preferences(self, mock_embedding_client):
        """Test HybridRAGDependencies has default search preferences."""
        deps = HybridRAGDependencies(
            embedding_client=mock_embedding_client,
            embedding_model="test-model"
        )
        assert deps.search_preferences is not None
        assert deps.search_preferences.use_vector is True


# =============================================================================
# Message History Tests
# =============================================================================

class TestMessageHistory:
    """Test message history and context."""

    @pytest.mark.asyncio
    async def test_messages_recorded(self, test_dependencies):
        """Test that messages are recorded during execution."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = await agent.run(
                "Test message",
                deps=test_dependencies
            )

            messages = result.all_messages()
            assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_new_messages_available(self, test_dependencies):
        """Test that new messages are available after execution."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = await agent.run(
                "Another test",
                deps=test_dependencies
            )

            new_msgs = result.new_messages()
            assert len(new_msgs) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test agent error handling capabilities."""

    @pytest.mark.asyncio
    async def test_handles_tool_exception(self, test_dependencies):
        """Test agent handles tool exceptions gracefully."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Database error")

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for content",
                    deps=test_dependencies
                )
                # Agent should still complete
                assert result.output is not None

    @pytest.mark.asyncio
    async def test_handles_embedding_exception(self, test_dependencies):
        """Test agent handles embedding generation errors."""
        # Make embedding client raise an error
        test_dependencies.embedding_client.embeddings.create.side_effect = Exception("API error")

        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock):
            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for content",
                    deps=test_dependencies
                )
                # Agent should handle the error
                assert result.output is not None


# =============================================================================
# Streaming Capability Tests
# =============================================================================

class TestStreamingCapability:
    """Test agent streaming capabilities."""

    @pytest.mark.asyncio
    async def test_iter_method_available(self, test_dependencies):
        """Test that iter method is available for streaming."""
        test_model = TestModel()

        with agent.override(model=test_model):
            async with agent.iter("Test", deps=test_dependencies) as run:
                nodes = []
                async for node in run:
                    nodes.append(node)

                assert len(nodes) > 0

    @pytest.mark.asyncio
    async def test_streaming_produces_result(self, test_dependencies):
        """Test that streaming execution produces a result."""
        test_model = TestModel()

        with agent.override(model=test_model):
            async with agent.iter("Hello", deps=test_dependencies) as run:
                async for _ in run:
                    pass

                assert run.result is not None
                assert run.result.output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
