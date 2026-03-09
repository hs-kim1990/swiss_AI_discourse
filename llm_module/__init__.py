"""
LLM Module for Document Classification
A framework for classifying JSON documents using local LLM models via vllm server.
"""

from .document_loader import DocumentLoader
from .llm_classifier import LLMClassifier
from .config import Config

__version__ = "0.1.0"
__all__ = ["DocumentLoader", "LLMClassifier", "Config"]
