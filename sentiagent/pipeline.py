"""
pipeline.py
-----------
Orchestrates the five-agent SentiAgent pipeline end to end:

    Input Review (X)
        -> Context Agent (C_A)
        -> Knowledge Agent (K_A)
        -> Sentiment Agent (S_A)
        -> Reasoning Agent (R_A)
        -> Explainability Agent (E_A)
        -> Output (final sentiment, confidence, explanation)

This mirrors the sequential architecture in Figure 1 of the paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .agents import (
    ContextAgent,
    ExplainabilityAgent,
    KnowledgeAgent,
    ReasoningAgent,
    SentimentAgent,
)
from .knowledge_sources import ConceptNetRetriever
from .llm_client import LLMClient


@dataclass
class SentiAgentResult:
    """Final output plus every intermediate agent output, for auditability."""

    review_text: str
    final_label: str
    confidence: float
    explanation: str
    context_output: Dict[str, Any] = field(default_factory=dict)
    knowledge_output: Dict[str, Any] = field(default_factory=dict)
    sentiment_output: Dict[str, Any] = field(default_factory=dict)
    reasoning_output: Dict[str, Any] = field(default_factory=dict)
    explainability_output: Dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.review_text,
            "output": {
                "final_label": self.final_label,
                "confidence": self.confidence,
                "explanation": self.explanation,
            },
            "trace": {
                "context_agent": self.context_output,
                "knowledge_agent": self.knowledge_output,
                "sentiment_agent": self.sentiment_output,
                "reasoning_agent": self.reasoning_output,
                "explainability_agent": self.explainability_output,
            },
            "latency_seconds": self.latency_seconds,
        }


class SentiAgentPipeline:
    """
    High-level entry point. Construct once, call `.run(text)` per review.

    Parameters
    ----------
    model : str
        LLM model identifier shared by all five agents.
    backend : str
        "anthropic" (default) or "openai".
    api_key : Optional[str]
        Overrides the provider's default environment-variable API key.
    knowledge_retriever : Optional[ConceptNetRetriever]
        Custom external-knowledge retriever for the Knowledge Agent.
        Defaults to a ConceptNet-backed retriever.
    verbose : bool
        If True, prints each agent's stage as it runs.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        backend: str = "anthropic",
        api_key: Optional[str] = None,
        knowledge_retriever: Optional[ConceptNetRetriever] = None,
        verbose: bool = False,
    ) -> None:
        llm = LLMClient(model=model, backend=backend, api_key=api_key)

        self.context_agent = ContextAgent(llm)
        self.knowledge_agent = KnowledgeAgent(llm, retriever=knowledge_retriever)
        self.sentiment_agent = SentimentAgent(llm)
        self.reasoning_agent = ReasoningAgent(llm)
        self.explainability_agent = ExplainabilityAgent(llm)
        self.verbose = verbose

    def run(self, review_text: str) -> SentiAgentResult:
        """Run a single review through the full five-agent pipeline."""
        start = time.time()

        self._log("1/5 Context Agent...")
        o_c = self.context_agent.run(review_text)

        self._log("2/5 Knowledge Agent...")
        o_k = self.knowledge_agent.run(review_text, o_c)

        self._log("3/5 Sentiment Agent...")
        o_s = self.sentiment_agent.run(review_text, o_c, o_k)

        self._log("4/5 Reasoning Agent...")
        o_r = self.reasoning_agent.run(review_text, o_c, o_k, o_s)

        self._log("5/5 Explainability Agent...")
        o_e = self.explainability_agent.run(review_text, o_c, o_k, o_s, o_r)

        elapsed = time.time() - start
        self._log(f"Done in {elapsed:.2f}s")

        return SentiAgentResult(
            review_text=review_text,
            final_label=o_e["final_label"],
            confidence=o_e["confidence"],
            explanation=o_e["explanation"],
            context_output=o_c,
            knowledge_output=o_k,
            sentiment_output=o_s,
            reasoning_output=o_r,
            explainability_output=o_e,
            latency_seconds=elapsed,
        )

    def run_batch(self, review_texts: list) -> list:
        """Run the pipeline over a list of reviews, returning a list of results."""
        return [self.run(text) for text in review_texts]

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[SentiAgent] {message}")
