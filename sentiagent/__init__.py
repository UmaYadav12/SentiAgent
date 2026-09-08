"""
SentiAgent: An Agentic LLM Framework for Context-Aware and Explainable Sentiment Analysis.

Public API:
    from sentiagent import SentiAgentPipeline

    pipeline = SentiAgentPipeline()
    result = pipeline.run("The phone camera is excellent, but the battery drains quickly.")
    print(result.to_dict())
"""

from .pipeline import SentiAgentPipeline, SentiAgentResult
from .agents import (
    ContextAgent,
    KnowledgeAgent,
    SentimentAgent,
    ReasoningAgent,
    ExplainabilityAgent,
)
from .llm_client import LLMClient

__all__ = [
    "SentiAgentPipeline",
    "SentiAgentResult",
    "ContextAgent",
    "KnowledgeAgent",
    "SentimentAgent",
    "ReasoningAgent",
    "ExplainabilityAgent",
    "LLMClient",
]

__version__ = "1.0.0"
