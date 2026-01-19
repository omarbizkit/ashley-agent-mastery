"""Configuration management using pydantic-settings."""

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

    # Embedding Configuration
    embedding_api_key: str = Field(..., description="API key for embeddings")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    embedding_base_url: str = Field(default="https://api.openai.com/v1", description="Embedding API base URL")

    # Neo4j Configuration (for Graphiti knowledge graph)
    neo4j_uri: str = Field(..., description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    # Feature Flags
    use_graph_search: bool = Field(default=True, description="Enable graph search (requires Neo4j)")

    # Application Configuration
    log_level: str = Field(default="INFO")


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "llm_api_key" in str(e).lower():
            error_msg += "\nMake sure to set LLM_API_KEY in your .env file"
        raise ValueError(error_msg) from e


settings = load_settings()
