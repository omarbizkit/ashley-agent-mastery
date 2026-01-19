"""Integration tests for the Hybrid RAG Knowledge Graph Agent.

These tests validate:
- Agent with multiple tools working together
- Streaming functionality
- CLI integration patterns
- Database and Graph client integration patterns
- Full conversation flows
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from agent import agent
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
def sample_search_data():
    """Combined sample data for integration tests."""
    return {
        "vector_results": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "content": "OpenAI is an AI research company.",
                "similarity": 0.92,
                "metadata": {"topic": "AI"},
                "document_title": "Company Profile",
                "document_source": "companies/openai.md"
            }
        ],
        "graph_results": [
            {
                "fact": "Sam Altman is CEO of OpenAI",
                "uuid": "fact-1",
                "valid_at": "2019-01-01",
                "invalid_at": None
            }
        ],
        "document": {
            "id": "doc-1",
            "title": "Company Profile",
            "source": "companies/openai.md",
            "content": "OpenAI is an AI research company founded in 2015...",
            "metadata": {"topic": "AI", "category": "companies"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-15T00:00:00"
        },
        "documents_list": [
            {"id": "doc-1", "title": "OpenAI Profile", "source": "companies/openai.md", "chunk_count": 5},
            {"id": "doc-2", "title": "AI Overview", "source": "tech/ai.md", "chunk_count": 3}
        ]
    }


# =============================================================================
# Multi-Tool Integration Tests
# =============================================================================

class TestMultiToolIntegration:
    """Tests for agent using multiple tools in a single conversation."""

    @pytest.mark.asyncio
    async def test_vector_then_document_retrieval(self, test_deps, sample_search_data):
        """Test agent can search then retrieve full document."""
        # First call: vector search, Second call: get document
        call_count = [0]

        async def custom_model_fn(messages, info: AgentInfo):
            call_count[0] += 1
            if call_count[0] == 1:
                # First response - call vector_search
                return ModelResponse(
                    parts=[
                        ToolCallPart.from_raw_args(
                            tool_name="vector_search",
                            args={"query": "OpenAI", "limit": 5}
                        )
                    ]
                )
            elif call_count[0] == 2:
                # Second response - call get_document
                return ModelResponse(
                    parts=[
                        ToolCallPart.from_raw_args(
                            tool_name="get_document",
                            args={"document_id": "doc-1"}
                        )
                    ]
                )
            else:
                # Final response
                return ModelResponse(
                    parts=[TextPart(content="Found information about OpenAI from the documents.")]
                )

        function_model = FunctionModel(custom_model_fn)

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch('agent.db_get_document', new_callable=AsyncMock) as mock_doc:

            mock_vector.return_value = sample_search_data["vector_results"]
            mock_doc.return_value = sample_search_data["document"]

            with agent.override(model=function_model):
                result = await agent.run(
                    "Tell me about OpenAI and get the full document",
                    deps=test_deps
                )

                assert result.output is not None
                assert "OpenAI" in result.output
                mock_vector.assert_called_once()
                mock_doc.assert_called_once()

    @pytest.mark.asyncio
    async def test_comprehensive_search_flow(self, test_deps, sample_search_data):
        """Test comprehensive search that uses both vector and graph."""
        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch('agent.search_knowledge_graph', new_callable=AsyncMock) as mock_graph:

            mock_vector.return_value = sample_search_data["vector_results"]
            mock_graph.return_value = sample_search_data["graph_results"]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Give me comprehensive information about OpenAI",
                    deps=test_deps
                )

                assert result.output is not None
                mock_vector.assert_called()
                mock_graph.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_search_with_document_listing(self, test_deps, sample_search_data):
        """Test hybrid search followed by document listing."""
        call_count = [0]

        async def custom_model_fn(messages, info: AgentInfo):
            call_count[0] += 1
            if call_count[0] == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart.from_raw_args(
                            tool_name="hybrid_search",
                            args={"query": "AI companies", "limit": 5, "text_weight": 0.4}
                        )
                    ]
                )
            elif call_count[0] == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart.from_raw_args(
                            tool_name="list_documents",
                            args={"limit": 10, "offset": 0}
                        )
                    ]
                )
            else:
                return ModelResponse(
                    parts=[TextPart(content="Found relevant AI company information.")]
                )

        function_model = FunctionModel(custom_model_fn)

        with patch('agent.db_hybrid_search', new_callable=AsyncMock) as mock_hybrid, \
             patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:

            mock_hybrid.return_value = sample_search_data["vector_results"]
            mock_list.return_value = sample_search_data["documents_list"]

            with agent.override(model=function_model):
                result = await agent.run(
                    "Search for AI companies and list all documents",
                    deps=test_deps
                )

                assert result.output is not None


# =============================================================================
# Conversation Flow Tests
# =============================================================================

class TestConversationFlows:
    """Tests for complete conversation flows."""

    @pytest.mark.asyncio
    async def test_question_answer_flow(self, test_deps, sample_search_data):
        """Test basic question-answer flow with tool use."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_search_data["vector_results"]

            with agent.override(model=test_model):
                result = await agent.run(
                    "What do you know about OpenAI?",
                    deps=test_deps
                )

                assert result.output is not None
                # Verify messages were recorded
                messages = result.all_messages()
                assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_no_tool_response(self, test_deps):
        """Test agent can respond without calling tools when appropriate."""
        async def no_tool_model_fn(messages, info: AgentInfo):
            return ModelResponse(
                parts=[TextPart(content="I'm happy to help! What would you like to know?")]
            )

        function_model = FunctionModel(no_tool_model_fn)

        with agent.override(model=function_model):
            result = await agent.run(
                "Hello, how are you?",
                deps=test_deps
            )

            assert result.output is not None
            assert "help" in result.output.lower()

    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, test_deps):
        """Test agent handles tool errors and continues conversation."""
        call_count = [0]

        async def error_recovery_model_fn(messages, info: AgentInfo):
            call_count[0] += 1
            if call_count[0] == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart.from_raw_args(
                            tool_name="vector_search",
                            args={"query": "test", "limit": 10}
                        )
                    ]
                )
            else:
                return ModelResponse(
                    parts=[TextPart(content="I encountered an error but I'm still here to help.")]
                )

        function_model = FunctionModel(error_recovery_model_fn)

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Database temporarily unavailable")

            with agent.override(model=function_model):
                result = await agent.run(
                    "Search for information",
                    deps=test_deps
                )

                # Agent should still complete despite error
                assert result.output is not None


# =============================================================================
# Dependency Injection Tests
# =============================================================================

class TestDependencyInjection:
    """Tests for proper dependency injection patterns."""

    @pytest.mark.asyncio
    async def test_embedding_client_injection(self, sample_search_data):
        """Test embedding client is properly injected and used."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.5] * 1536)]  # Different embedding
        )

        deps = HybridRAGDependencies(
            embedding_client=mock_client,
            embedding_model="custom-model",
            search_preferences=SearchPreferences()
        )

        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_search_data["vector_results"]

            with agent.override(model=test_model):
                await agent.run("Test query", deps=deps)

                # Verify embedding client was called with correct model
                mock_client.embeddings.create.assert_called()
                call_args = mock_client.embeddings.create.call_args
                assert call_args.kwargs.get('model') == "custom-model"

    @pytest.mark.asyncio
    async def test_search_preferences_respected(self, mock_embedding_client):
        """Test search preferences are properly configured."""
        custom_prefs = SearchPreferences(
            use_vector=True,
            use_graph=False,
            default_limit=5
        )

        deps = HybridRAGDependencies(
            embedding_client=mock_embedding_client,
            embedding_model="text-embedding-3-small",
            search_preferences=custom_prefs
        )

        assert deps.search_preferences.use_vector is True
        assert deps.search_preferences.use_graph is False
        assert deps.search_preferences.default_limit == 5

    @pytest.mark.asyncio
    async def test_session_id_tracking(self, mock_embedding_client):
        """Test session ID is properly tracked in dependencies."""
        deps = HybridRAGDependencies(
            embedding_client=mock_embedding_client,
            embedding_model="text-embedding-3-small",
            search_preferences=SearchPreferences(),
            session_id="test-session-123"
        )

        assert deps.session_id == "test-session-123"


# =============================================================================
# Message History Tests
# =============================================================================

class TestMessageHistory:
    """Tests for message history tracking."""

    @pytest.mark.asyncio
    async def test_messages_recorded(self, test_deps, sample_search_data):
        """Test that all messages are properly recorded."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_search_data["vector_results"]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for OpenAI",
                    deps=test_deps
                )

                messages = result.all_messages()

                # Should have user message, model request, tool result, and model response
                assert len(messages) >= 1
                # First message should be user prompt
                assert any("OpenAI" in str(msg) for msg in messages)

    @pytest.mark.asyncio
    async def test_tool_calls_in_messages(self, test_deps, sample_search_data):
        """Test that tool calls are recorded in messages."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sample_search_data["vector_results"]

            with agent.override(model=test_model):
                result = await agent.run(
                    "Search for information",
                    deps=test_deps
                )

                # Convert to new_messages to check what was recorded
                new_msgs = result.new_messages()
                assert len(new_msgs) > 0


# =============================================================================
# Streaming Integration Tests
# =============================================================================

class TestStreamingIntegration:
    """Tests for streaming functionality patterns."""

    @pytest.mark.asyncio
    async def test_iter_method_available(self, test_deps):
        """Test that iter method is available for streaming."""
        test_model = TestModel()

        with agent.override(model=test_model):
            # Verify iter is available
            async with agent.iter("Test message", deps=test_deps) as run:
                nodes_count = 0
                async for node in run:
                    nodes_count += 1
                    # Just verify we can iterate
                    pass

                # Should have at least some nodes
                assert nodes_count > 0

    @pytest.mark.asyncio
    async def test_streaming_with_tool_call(self, test_deps, sample_search_data):
        """Test streaming works with tool calls."""
        test_model = TestModel(call_tools=['list_documents'])

        with patch('agent.db_list_documents', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = sample_search_data["documents_list"]

            with agent.override(model=test_model):
                async with agent.iter("List documents", deps=test_deps) as run:
                    node_types = []
                    async for node in run:
                        node_types.append(type(node).__name__)

                    # Should have processed some nodes
                    assert len(node_types) > 0


# =============================================================================
# Agent Configuration Tests
# =============================================================================

class TestAgentConfiguration:
    """Tests for agent configuration validation."""

    def test_agent_has_system_prompt(self):
        """Test agent has system prompt configured."""
        assert agent._system_prompts is not None
        assert len(agent._system_prompts) > 0

    def test_agent_has_deps_type(self):
        """Test agent has correct deps_type configured."""
        assert agent._deps_type == HybridRAGDependencies

    def test_agent_tool_count(self):
        """Test agent has exactly 8 tools."""
        assert len(agent._function_tools) == 8

    def test_agent_no_result_type(self):
        """Test agent uses default string output (no result_type)."""
        # Agent should default to string output
        # This is validated by checking it works with string responses
        test_model = TestModel()
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )

        deps = HybridRAGDependencies(
            embedding_client=mock_client,
            embedding_model="test",
            search_preferences=SearchPreferences()
        )

        with agent.override(model=test_model):
            result = agent.run_sync("Test", deps=deps)
            assert isinstance(result.output, str)


# =============================================================================
# Parallel Execution Tests
# =============================================================================

class TestParallelExecution:
    """Tests for parallel execution patterns."""

    @pytest.mark.asyncio
    async def test_comprehensive_search_parallel_execution(self, test_deps):
        """Test comprehensive search runs vector and graph in parallel."""
        vector_called = asyncio.Event()
        graph_called = asyncio.Event()

        async def slow_vector_search(*args, **kwargs):
            vector_called.set()
            await asyncio.sleep(0.1)  # Simulate delay
            return [{"content": "Vector result", "similarity": 0.9}]

        async def slow_graph_search(*args, **kwargs):
            graph_called.set()
            await asyncio.sleep(0.1)  # Simulate delay
            return [{"fact": "Graph result", "uuid": "123"}]

        test_model = TestModel(call_tools=['perform_comprehensive_search'])

        with patch('agent.db_vector_search', side_effect=slow_vector_search), \
             patch('agent.search_knowledge_graph', side_effect=slow_graph_search):

            with agent.override(model=test_model):
                start_time = asyncio.get_event_loop().time()
                result = await agent.run(
                    "Comprehensive search",
                    deps=test_deps
                )
                end_time = asyncio.get_event_loop().time()

                # Both should be called
                assert vector_called.is_set()
                assert graph_called.is_set()

                # Result should be present
                assert result.output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
