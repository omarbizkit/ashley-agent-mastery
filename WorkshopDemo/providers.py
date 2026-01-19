"""Flexible provider configuration for LLM and embedding models."""

from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from settings import settings


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
