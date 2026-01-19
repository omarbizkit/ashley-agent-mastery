"""Agent dependencies using dataclass pattern."""

from dataclasses import dataclass, field
from typing import Optional
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
