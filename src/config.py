"""Configuration for multi-model LLM setup"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class ModelConfig:
    """Configuration for multi-model setup"""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:27b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def get_model(
        self, provider: Optional[str] = None, temperature: float = 0.7
    ) -> BaseChatModel:
        """Get a configured model instance"""
        provider = provider or self.provider

        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=self.openai_model,
                temperature=temperature,
            )

        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return ChatAnthropic(
                api_key=self.anthropic_api_key,
                model=self.anthropic_model,
                temperature=temperature,
            )

        elif provider == "ollama":
            return ChatOllama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_all_models(self) -> dict[str, BaseChatModel]:
        """Get all available models"""
        models = {}

        # Try to initialize each model
        for provider in ["openai", "anthropic", "ollama"]:
            try:
                models[provider] = self.get_model(provider)
                print(f"✓ {provider.capitalize()} model initialized")
            except Exception as e:
                print(f"✗ {provider.capitalize()} model failed: {e}")

        return models


# Global config instance
config = ModelConfig()


def setup_and_get_models():
    """Setup environment and get all available models"""
    print("LangChain Multi-Model Examples")
    print("==============================")
    print(f"Primary provider: {config.provider}")

    # Get available models
    models = config.get_all_models()
    if not models:
        print("No models available! Please check your configuration.")
        return None

    print(f"\nAvailable models: {list(models.keys())}")
    return models
