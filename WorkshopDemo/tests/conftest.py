"""Test fixtures for the Hybrid RAG Knowledge Graph Agent.

This module provides shared fixtures used across all test modules:
- Mock embedding clients
- Test dependencies
- Sample search results
- Mock database responses
- Mock graph responses
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dependencies import HybridRAGDependencies, SearchPreferences


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_client():
    """Create mock OpenAI async embedding client.

    Returns a mock that simulates the OpenAI embeddings API,
    returning 1536-dimensional vectors.
    """
    client = AsyncMock()
    client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    return client


@pytest.fixture
def mock_embedding_client_custom():
    """Create mock embedding client that can be customized.

    Use this when you need to change the embedding response per test.
    """
    def _create_client(embedding=None):
        client = AsyncMock()
        if embedding is None:
            embedding = [0.1] * 1536
        client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=embedding)]
        )
        return client
    return _create_client


@pytest.fixture
def test_dependencies(mock_embedding_client):
    """Create standard test dependencies."""
    return HybridRAGDependencies(
        embedding_client=mock_embedding_client,
        embedding_model="text-embedding-3-small",
        search_preferences=SearchPreferences()
    )


@pytest.fixture
def test_deps(mock_embedding_client):
    """Alias for test_dependencies for shorter name."""
    return HybridRAGDependencies(
        embedding_client=mock_embedding_client,
        embedding_model="text-embedding-3-small",
        search_preferences=SearchPreferences()
    )


@pytest.fixture
def test_deps_with_session(mock_embedding_client):
    """Create test dependencies with a session ID."""
    return HybridRAGDependencies(
        embedding_client=mock_embedding_client,
        embedding_model="text-embedding-3-small",
        search_preferences=SearchPreferences(),
        session_id="test-session-123"
    )


@pytest.fixture
def custom_search_preferences():
    """Create custom search preferences."""
    return SearchPreferences(
        use_vector=True,
        use_graph=True,
        default_limit=25
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_vector_results():
    """Sample vector search results."""
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Artificial intelligence is transforming industries worldwide.",
            "similarity": 0.95,
            "metadata": {"topic": "AI", "category": "technology"},
            "document_title": "AI Overview",
            "document_source": "test/ai.md"
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "content": "Machine learning enables computers to learn from data patterns.",
            "similarity": 0.85,
            "metadata": {"topic": "ML", "category": "technology"},
            "document_title": "ML Basics",
            "document_source": "test/ml.md"
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "doc-1",
            "content": "Deep learning uses neural networks with multiple layers.",
            "similarity": 0.78,
            "metadata": {"topic": "AI", "category": "technology"},
            "document_title": "AI Overview",
            "document_source": "test/ai.md"
        }
    ]


@pytest.fixture
def sample_graph_results():
    """Sample graph search results."""
    return [
        {
            "fact": "OpenAI created ChatGPT in November 2022",
            "uuid": "fact-uuid-1",
            "valid_at": "2022-11-30",
            "invalid_at": None,
            "source_node_uuid": "node-openai"
        },
        {
            "fact": "Sam Altman is the CEO of OpenAI",
            "uuid": "fact-uuid-2",
            "valid_at": "2019-01-01",
            "invalid_at": None,
            "source_node_uuid": "node-sam"
        },
        {
            "fact": "OpenAI partners with Microsoft for cloud computing",
            "uuid": "fact-uuid-3",
            "valid_at": "2019-07-22",
            "invalid_at": None,
            "source_node_uuid": "node-partnership"
        }
    ]


@pytest.fixture
def sample_hybrid_results():
    """Sample hybrid search results."""
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Python is a versatile programming language used in AI and web development.",
            "combined_score": 0.88,
            "vector_similarity": 0.85,
            "text_similarity": 0.95,
            "metadata": {"topic": "programming", "language": "python"},
            "document_title": "Python Guide",
            "document_source": "test/python.md"
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-3",
            "content": "Python's simplicity makes it ideal for beginners and experts alike.",
            "combined_score": 0.75,
            "vector_similarity": 0.72,
            "text_similarity": 0.80,
            "metadata": {"topic": "programming", "language": "python"},
            "document_title": "Python Tutorial",
            "document_source": "test/python-tutorial.md"
        }
    ]


@pytest.fixture
def sample_document():
    """Sample document data."""
    return {
        "id": "doc-uuid-1",
        "title": "AI Overview",
        "source": "test/ai.md",
        "content": """Artificial Intelligence Overview

Artificial intelligence (AI) is transforming industries worldwide.
Machine learning, a subset of AI, enables computers to learn from data.
Deep learning uses neural networks with many layers for complex tasks.

Key applications include:
- Natural Language Processing
- Computer Vision
- Robotics
- Autonomous Vehicles

AI continues to evolve rapidly with new breakthroughs emerging regularly.""",
        "metadata": {"topic": "AI", "category": "technology", "author": "Test Author"},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-15T12:30:00"
    }


@pytest.fixture
def sample_documents_list():
    """Sample documents list for pagination testing."""
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
        },
        {
            "id": "doc-3",
            "title": "Python Guide",
            "source": "test/python.md",
            "metadata": {"topic": "programming"},
            "created_at": "2024-01-03T00:00:00",
            "updated_at": "2024-01-17T00:00:00",
            "chunk_count": 8
        },
        {
            "id": "doc-4",
            "title": "Cloud Computing",
            "source": "test/cloud.md",
            "metadata": {"topic": "cloud"},
            "created_at": "2024-01-04T00:00:00",
            "updated_at": "2024-01-18T00:00:00",
            "chunk_count": 4
        }
    ]


@pytest.fixture
def sample_entity_relationships():
    """Sample entity relationships data."""
    return {
        "central_entity": "OpenAI",
        "related_facts": [
            {
                "fact": "OpenAI created ChatGPT",
                "uuid": "fact-1",
                "valid_at": "2022-11-30"
            },
            {
                "fact": "OpenAI partners with Microsoft",
                "uuid": "fact-2",
                "valid_at": "2019-07-22"
            },
            {
                "fact": "Sam Altman leads OpenAI as CEO",
                "uuid": "fact-3",
                "valid_at": "2019-01-01"
            }
        ],
        "search_method": "graphiti_semantic_search"
    }


@pytest.fixture
def sample_entity_timeline():
    """Sample entity timeline data."""
    return [
        {
            "fact": "Company founded in San Francisco",
            "uuid": "timeline-1",
            "valid_at": "2015-12-11",
            "invalid_at": None
        },
        {
            "fact": "Released GPT-2 model",
            "uuid": "timeline-2",
            "valid_at": "2019-02-14",
            "invalid_at": None
        },
        {
            "fact": "Released GPT-3 model",
            "uuid": "timeline-3",
            "valid_at": "2020-06-11",
            "invalid_at": None
        },
        {
            "fact": "Launched ChatGPT product",
            "uuid": "timeline-4",
            "valid_at": "2022-11-30",
            "invalid_at": None
        }
    ]


@pytest.fixture
def sample_comprehensive_results(sample_vector_results, sample_graph_results):
    """Sample comprehensive search results combining vector and graph."""
    return {
        "vector_results": sample_vector_results,
        "graph_results": sample_graph_results
    }


# =============================================================================
# Mock Object Factories
# =============================================================================

@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


@pytest.fixture
def mock_graph_client():
    """Create mock Graphiti client."""
    client = AsyncMock()
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.get_related_entities = AsyncMock(return_value={})
    client.get_entity_timeline = AsyncMock(return_value=[])
    return client


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# =============================================================================
# Pytest Async Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
