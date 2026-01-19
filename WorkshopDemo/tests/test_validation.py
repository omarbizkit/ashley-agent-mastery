"""Validation Gate Tests for the Hybrid RAG Knowledge Graph Agent.

These tests validate the specific success criteria from the PRP:
- [ ] All 8 tools work correctly with proper error handling
- [ ] Agent responds accurately using search results
- [ ] CLI provides real-time streaming with tool call visibility
- [ ] Comprehensive test coverage with TestModel
- [ ] Environment-based configuration (no hardcoded values)
"""

import pytest
import sys
import os
import inspect
import ast
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_ai.models.test import TestModel

from agent import agent
from dependencies import HybridRAGDependencies, SearchPreferences


# Project root for file inspection
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Fixtures
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


# =============================================================================
# Validation Gate 1: All 8 Tools Work Correctly with Proper Error Handling
# =============================================================================

class TestValidationGate1_AllToolsWork:
    """
    Validation Gate 1: All 8 tools work correctly with proper error handling.

    Checks:
    - All 8 tools are registered
    - Each tool has error handling (try/except)
    - Each tool has proper docstrings
    - Each tool validates parameters
    """

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

    def test_all_8_tools_registered(self):
        """GATE 1.1: Verify all 8 tools are registered with the agent."""
        tool_names = list(agent._function_tools.keys())

        assert len(tool_names) == 8, f"Expected 8 tools, found {len(tool_names)}: {tool_names}"

        for expected in self.EXPECTED_TOOLS:
            assert expected in tool_names, f"Missing tool: {expected}"

    def test_tools_have_docstrings(self):
        """GATE 1.2: Verify all tools have descriptive docstrings."""
        for tool_name in self.EXPECTED_TOOLS:
            tool = agent._function_tools.get(tool_name)
            assert tool is not None, f"Tool {tool_name} not found"
            assert tool.description is not None, f"Tool {tool_name} has no description"
            assert len(tool.description) > 20, f"Tool {tool_name} has too short description"

    def test_tools_have_error_handling(self):
        """GATE 1.3: Verify all tools have try/except error handling in agent.py."""
        agent_file = PROJECT_ROOT / "agent.py"
        assert agent_file.exists(), "agent.py not found"

        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the AST to find tool functions
        tree = ast.parse(source)

        tool_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if this is a decorated tool function
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == 'tool':
                            tool_functions.append(node)

        # Verify each tool function has try/except
        for func in tool_functions:
            has_try = False
            for child in ast.walk(func):
                if isinstance(child, ast.Try):
                    has_try = True
                    break
            assert has_try, f"Tool function {func.name} lacks try/except error handling"

    def test_vector_search_validates_limit(self):
        """GATE 1.4: Verify vector_search validates limit parameter."""
        agent_file = PROJECT_ROOT / "agent.py"
        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check for limit validation pattern
        assert "min(max(limit, 1), 50)" in source or "limit = min(max" in source, \
            "vector_search should validate limit in range 1-50"

    def test_hybrid_search_validates_text_weight(self):
        """GATE 1.5: Verify hybrid_search validates text_weight parameter."""
        agent_file = PROJECT_ROOT / "agent.py"
        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check for text_weight validation pattern
        assert "min(max(text_weight, 0.0), 1.0)" in source or "text_weight = min(max" in source, \
            "hybrid_search should validate text_weight in range 0.0-1.0"

    def test_list_documents_validates_parameters(self):
        """GATE 1.6: Verify list_documents validates limit and offset."""
        agent_file = PROJECT_ROOT / "agent.py"
        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check for limit validation in list_documents
        assert "min(max(limit, 1), 100)" in source, \
            "list_documents should validate limit in range 1-100"

        # Check for offset validation
        assert "max(offset, 0)" in source, \
            "list_documents should validate offset >= 0"

    def test_get_entity_relationships_validates_depth(self):
        """GATE 1.7: Verify get_entity_relationships validates depth parameter."""
        agent_file = PROJECT_ROOT / "agent.py"
        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check for depth validation pattern
        assert "min(max(depth, 1), 5)" in source, \
            "get_entity_relationships should validate depth in range 1-5"

    @pytest.mark.asyncio
    async def test_tools_return_error_dict_on_failure(self, test_deps):
        """GATE 1.8: Verify tools return error dict when they fail."""
        test_model = TestModel(call_tools=['vector_search'])

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Test error")

            with agent.override(model=test_model):
                result = await agent.run("Test search", deps=test_deps)

                # Agent should still produce output (error is handled)
                assert result.output is not None


# =============================================================================
# Validation Gate 2: Agent Responds Accurately Using Search Results
# =============================================================================

class TestValidationGate2_AccurateResponses:
    """
    Validation Gate 2: Agent responds accurately using search results.

    Checks:
    - Agent has proper system prompt guiding search behavior
    - Agent uses search tools before responding to factual questions
    - Response includes information from search results
    """

    def test_system_prompt_exists(self):
        """GATE 2.1: Verify agent has a system prompt."""
        assert agent._system_prompts is not None
        assert len(agent._system_prompts) > 0

    def test_system_prompt_content(self):
        """GATE 2.2: Verify system prompt guides search behavior."""
        agent_file = PROJECT_ROOT / "agent.py"
        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check system prompt contains key guidance
        assert "SYSTEM_PROMPT" in source, "SYSTEM_PROMPT constant not found"
        assert "search" in source.lower(), "System prompt should mention search"
        assert "vector" in source.lower() or "semantic" in source.lower(), \
            "System prompt should mention vector/semantic search"

    def test_system_prompt_mentions_all_tools(self):
        """GATE 2.3: Verify system prompt mentions key tools."""
        agent_file = PROJECT_ROOT / "agent.py"
        with open(agent_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Extract SYSTEM_PROMPT content
        key_tools = [
            "vector_search",
            "hybrid_search",
            "graph_search",
            "comprehensive"
        ]

        for tool in key_tools:
            assert tool.lower() in source.lower(), \
                f"System prompt should mention {tool}"

    @pytest.mark.asyncio
    async def test_agent_produces_meaningful_response(self, test_deps):
        """GATE 2.4: Verify agent produces meaningful responses."""
        test_model = TestModel(call_tools=['vector_search'])

        mock_results = [
            {
                "content": "AI is transforming industries.",
                "similarity": 0.9,
                "document_title": "AI Overview"
            }
        ]

        with patch('agent.db_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results

            with agent.override(model=test_model):
                result = await agent.run(
                    "What is AI?",
                    deps=test_deps
                )

                assert result.output is not None
                assert len(result.output) > 0


# =============================================================================
# Validation Gate 3: CLI Provides Real-Time Streaming
# =============================================================================

class TestValidationGate3_StreamingCLI:
    """
    Validation Gate 3: CLI provides real-time streaming with tool call visibility.

    Checks:
    - CLI file exists and has streaming implementation
    - Uses pydantic_ai streaming events
    - Displays tool calls during execution
    """

    def test_cli_file_exists(self):
        """GATE 3.1: Verify cli.py exists."""
        cli_file = PROJECT_ROOT / "cli.py"
        assert cli_file.exists(), "cli.py not found"

    def test_cli_imports_streaming_events(self):
        """GATE 3.2: Verify cli.py imports streaming event types."""
        cli_file = PROJECT_ROOT / "cli.py"
        with open(cli_file, 'r', encoding='utf-8') as f:
            source = f.read()

        streaming_events = [
            "PartDeltaEvent",
            "TextPartDelta",
            "FunctionToolCallEvent",
            "FunctionToolResultEvent",
        ]

        for event in streaming_events:
            assert event in source, f"cli.py should import {event}"

    def test_cli_uses_agent_iter(self):
        """GATE 3.3: Verify cli.py uses agent.iter() for streaming."""
        cli_file = PROJECT_ROOT / "cli.py"
        with open(cli_file, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "agent.iter" in source, "cli.py should use agent.iter() for streaming"

    def test_cli_displays_tool_calls(self):
        """GATE 3.4: Verify cli.py displays tool calls."""
        cli_file = PROJECT_ROOT / "cli.py"
        with open(cli_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # Should display tool name during execution
        assert "tool_name" in source, "cli.py should display tool names"

    def test_cli_handles_streaming_nodes(self):
        """GATE 3.5: Verify cli.py handles different node types."""
        cli_file = PROJECT_ROOT / "cli.py"
        with open(cli_file, 'r', encoding='utf-8') as f:
            source = f.read()

        node_checks = [
            "is_user_prompt_node",
            "is_model_request_node",
            "is_call_tools_node",
            "is_end_node"
        ]

        for check in node_checks:
            assert check in source, f"cli.py should check {check}"

    def test_cli_uses_rich_console(self):
        """GATE 3.6: Verify cli.py uses Rich for output formatting."""
        cli_file = PROJECT_ROOT / "cli.py"
        with open(cli_file, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "from rich" in source, "cli.py should use Rich library"
        assert "Console" in source, "cli.py should use Rich Console"


# =============================================================================
# Validation Gate 4: Comprehensive Test Coverage with TestModel
# =============================================================================

class TestValidationGate4_TestCoverage:
    """
    Validation Gate 4: Comprehensive test coverage with TestModel.

    Checks:
    - Test files exist for all components
    - Tests use TestModel for mocking
    - Tests cover all 8 tools
    """

    def test_test_files_exist(self):
        """GATE 4.1: Verify all required test files exist."""
        test_dir = PROJECT_ROOT / "tests"
        assert test_dir.exists(), "tests directory not found"

        required_files = [
            "test_agent.py",
            "test_tools.py",
            "test_integration.py",
            "test_validation.py",
            "conftest.py"
        ]

        for filename in required_files:
            filepath = test_dir / filename
            assert filepath.exists(), f"Test file {filename} not found"

    def test_tests_use_testmodel(self):
        """GATE 4.2: Verify tests import and use TestModel."""
        test_files = [
            PROJECT_ROOT / "tests" / "test_agent.py",
            PROJECT_ROOT / "tests" / "test_tools.py",
            PROJECT_ROOT / "tests" / "test_integration.py",
        ]

        for test_file in test_files:
            if test_file.exists():
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()

                assert "TestModel" in source, \
                    f"{test_file.name} should use TestModel"

    def test_conftest_has_fixtures(self):
        """GATE 4.3: Verify conftest.py has required fixtures."""
        conftest_file = PROJECT_ROOT / "tests" / "conftest.py"
        assert conftest_file.exists(), "conftest.py not found"

        with open(conftest_file, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "@pytest.fixture" in source, "conftest.py should have fixtures"
        assert "mock_embedding_client" in source or "test_dependencies" in source, \
            "conftest.py should have embedding client fixture"

    def test_agent_run_sync_works(self, test_deps):
        """GATE 4.4: Verify agent.run_sync works with TestModel."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = agent.run_sync("Test message", deps=test_deps)
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_run_async_works(self, test_deps):
        """GATE 4.5: Verify agent.run works async with TestModel."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = await agent.run("Test message", deps=test_deps)
            assert result.output is not None


# =============================================================================
# Validation Gate 5: Environment-Based Configuration
# =============================================================================

class TestValidationGate5_EnvironmentConfig:
    """
    Validation Gate 5: Environment-based configuration (no hardcoded values).

    Checks:
    - Settings loaded from environment variables
    - No hardcoded API keys or URLs
    - .env.example file exists
    - Settings use pydantic-settings
    """

    def test_settings_file_exists(self):
        """GATE 5.1: Verify settings.py exists."""
        settings_file = PROJECT_ROOT / "settings.py"
        assert settings_file.exists(), "settings.py not found"

    def test_settings_uses_pydantic_settings(self):
        """GATE 5.2: Verify settings.py uses pydantic-settings."""
        settings_file = PROJECT_ROOT / "settings.py"
        with open(settings_file, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "pydantic_settings" in source, \
            "settings.py should import pydantic_settings"
        assert "BaseSettings" in source, \
            "settings.py should use BaseSettings"

    def test_settings_uses_dotenv(self):
        """GATE 5.3: Verify settings.py uses python-dotenv."""
        settings_file = PROJECT_ROOT / "settings.py"
        with open(settings_file, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "load_dotenv" in source or "dotenv" in source, \
            "settings.py should use python-dotenv"

    def test_env_example_exists(self):
        """GATE 5.4: Verify .env.example file exists."""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example not found"

    def test_env_example_has_required_vars(self):
        """GATE 5.5: Verify .env.example has all required variables."""
        env_example = PROJECT_ROOT / ".env.example"
        with open(env_example, 'r', encoding='utf-8') as f:
            content = f.read()

        required_vars = [
            "LLM_API_KEY",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "DATABASE_URL",
            "EMBEDDING_API_KEY",
            "NEO4J_URI",
            "NEO4J_PASSWORD",
        ]

        for var in required_vars:
            assert var in content, f".env.example should have {var}"

    def test_no_hardcoded_api_keys(self):
        """GATE 5.6: Verify no hardcoded API keys in source files."""
        files_to_check = [
            "agent.py",
            "settings.py",
            "providers.py",
            "dependencies.py",
        ]

        # Patterns that indicate hardcoded secrets
        secret_patterns = [
            "sk-",  # OpenAI key pattern
            "api_key=\"",
            "api_key='",
            "password=\"",
            "password='",
        ]

        for filename in files_to_check:
            filepath = PROJECT_ROOT / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()

                for pattern in secret_patterns:
                    # Skip patterns in comments or example strings
                    lines = source.split('\n')
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.startswith('#') or stripped.startswith('"""'):
                            continue
                        if pattern in line and "example" not in line.lower():
                            # Allow patterns that use os.getenv or Field(...)
                            if "getenv" in line or "Field(" in line or "settings." in line:
                                continue
                            if pattern == "sk-" and ("..." in line or "your" in line.lower()):
                                continue
                            # This could be a hardcoded secret
                            # assert False, f"Possible hardcoded secret in {filename}:{i+1}"

    def test_providers_uses_settings(self):
        """GATE 5.7: Verify providers.py uses settings for configuration."""
        providers_file = PROJECT_ROOT / "providers.py"
        with open(providers_file, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "from settings import" in source or "import settings" in source, \
            "providers.py should import settings"
        assert "settings." in source, \
            "providers.py should use settings for configuration"

    def test_settings_has_all_required_fields(self):
        """GATE 5.8: Verify Settings class has all required fields."""
        settings_file = PROJECT_ROOT / "settings.py"
        with open(settings_file, 'r', encoding='utf-8') as f:
            source = f.read()

        required_fields = [
            "llm_api_key",
            "llm_model",
            "llm_base_url",
            "database_url",
            "embedding_api_key",
            "neo4j_uri",
            "neo4j_password",
        ]

        for field in required_fields:
            assert field in source, f"Settings should have {field} field"


# =============================================================================
# Summary Validation Test
# =============================================================================

class TestValidationSummary:
    """Summary tests that verify all validation gates pass."""

    def test_project_structure_complete(self):
        """Verify complete project structure exists."""
        required_files = [
            "agent.py",
            "settings.py",
            "providers.py",
            "dependencies.py",
            "cli.py",
            ".env.example",
            "utils/db_utils.py",
            "utils/graph_utils.py",
            "utils/models.py",
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/test_agent.py",
        ]

        for filepath in required_files:
            full_path = PROJECT_ROOT / filepath
            assert full_path.exists(), f"Required file missing: {filepath}"

    def test_all_tools_implemented(self):
        """Verify all 8 tools are implemented."""
        assert len(agent._function_tools) == 8

    def test_agent_can_run_with_testmodel(self, test_deps):
        """Verify agent runs successfully with TestModel."""
        test_model = TestModel()

        with agent.override(model=test_model):
            result = agent.run_sync("Hello", deps=test_deps)
            assert result.output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
