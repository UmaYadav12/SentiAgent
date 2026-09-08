"""
agents.py
---------
Implements the five agents of the SentiAgent framework:

    1. ContextAgent        (C_A)
    2. KnowledgeAgent       (K_A)
    3. SentimentAgent       (S_A)
    4. ReasoningAgent       (R_A)
    5. ExplainabilityAgent  (E_A)

Each agent's system prompt, inputs, and structured JSON output schema
follow Figure 1 of the paper exactly. Agents are stateless: they take
explicit inputs and return a JSON-serializable dict, which the
SentiAgentPipeline threads through the chain.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .knowledge_sources import ConceptNetRetriever
from .llm_client import LLMClient

ALLOWED_LABELS = {"Positive", "Negative", "Neutral", "Mixed"}


class BaseAgent:
    """Shared plumbing for all agents: holds an LLMClient and a fixed system prompt."""

    system_prompt: str = ""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _call(self, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
        return self.llm.complete_json(self.system_prompt, user_content)


# --------------------------------------------------------------------------- #
# 1. Context Agent
# --------------------------------------------------------------------------- #
class ContextAgent(BaseAgent):
    """
    Role & Objective: Understand the input review in detail, identify key
    entities, aspects, sentiment-bearing expressions, context, negation and
    contrast.
    """

    system_prompt = (
        "You are the Context Agent in a multi-agent sentiment analysis "
        "framework. Analyze the supplied text to identify the principal "
        "entities, sentiment-bearing aspects, contextual relationships, "
        "negation, contrast expressions, and relevant evidence. Do not "
        "produce the final overall sentiment classification. Return a "
        "structured representation containing the identified entities, "
        "aspects, contextual information, and supporting textual evidence. "
        "Respond as JSON with exactly these keys: "
        '{"entities": [string], "aspects": [{"aspect": string, "evidence": string}], '
        '"context": string, "negation": [string], "evidence": [string]}'
    )

    def run(self, review_text: str) -> Dict[str, Any]:
        """Input: X (raw review text). Output: O_C (structured JSON)."""
        result = self._call({"X": review_text})
        result.setdefault("entities", [])
        result.setdefault("aspects", [])
        result.setdefault("context", "")
        result.setdefault("negation", [])
        result.setdefault("evidence", [])
        return result


# --------------------------------------------------------------------------- #
# 2. Knowledge Agent
# --------------------------------------------------------------------------- #
class KnowledgeAgent(BaseAgent):
    """
    Role & Objective: Enrich the contextual representation using external
    commonsense knowledge (e.g. ConceptNet, COMET) relevant to the
    identified aspects.
    """

    system_prompt = (
        "You are the Knowledge Agent in a multi-agent sentiment analysis "
        "framework. Using the input text, contextual representation, and "
        "retrieved external knowledge, identify commonsense concepts and "
        "relationships relevant to the sentiment-bearing aspects. Restrict "
        "the retrieved knowledge to information that can support "
        "interpretation of the input. Do not independently determine the "
        "final sentiment label and do not introduce unsupported facts. "
        "Return the relevant concepts, relationships, and their semantic "
        "relevance in structured form. Respond as JSON with exactly these "
        'keys: {"concepts": [string], "relations": [{"relation": string, '
        '"source": string}], "relevance": string}'
    )

    def __init__(self, llm_client: LLMClient, retriever: Optional[ConceptNetRetriever] = None):
        super().__init__(llm_client)
        self.retriever = retriever or ConceptNetRetriever()

    def run(self, review_text: str, context_output: Dict[str, Any]) -> Dict[str, Any]:
        """Input: X, O_C, K (external knowledge). Output: O_K (structured JSON)."""
        aspect_terms = [a.get("aspect", "") for a in context_output.get("aspects", [])]
        external_knowledge = self.retriever.retrieve(aspect_terms) if aspect_terms else {}

        result = self._call(
            {
                "X": review_text,
                "O_C": context_output,
                "K": external_knowledge,
            }
        )
        result.setdefault("concepts", [])
        result.setdefault("relations", [])
        result.setdefault("relevance", "")
        result["raw_external_knowledge"] = external_knowledge
        return result


# --------------------------------------------------------------------------- #
# 3. Sentiment Agent
# --------------------------------------------------------------------------- #
class SentimentAgent(BaseAgent):
    """
    Role & Objective: Infer aspect-level sentiment and initial overall
    sentiment label with confidence using the context and enriched
    knowledge.
    """

    system_prompt = (
        "You are the Sentiment Agent in a multi-agent sentiment analysis "
        "framework. Determine the sentiment expressed in the input using "
        "the original text, contextual representation, aspect-level "
        "evidence, and external knowledge. First determine sentiment at "
        "the aspect level and then infer the overall sentiment. The "
        "allowed sentiment classes are Positive, Negative, Neutral, and "
        "Mixed. Provide the predicted class, confidence, and evidence "
        "supporting the prediction. Do not generate the final user-facing "
        "explanation. Respond as JSON with exactly these keys: "
        '{"aspect_sentiments": [{"aspect": string, "sentiment": string}], '
        '"initial_label": string, "confidence": number, "evidence": [string]}'
    )

    def run(
        self,
        review_text: str,
        context_output: Dict[str, Any],
        knowledge_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Input: X, O_C, O_K. Output: O_S (structured JSON)."""
        result = self._call(
            {"X": review_text, "O_C": context_output, "O_K": knowledge_output}
        )
        result.setdefault("aspect_sentiments", [])
        result.setdefault("initial_label", "Neutral")
        result.setdefault("confidence", 0.5)
        result.setdefault("evidence", [])
        result["initial_label"] = _coerce_label(result["initial_label"])
        result["confidence"] = _clamp01(result["confidence"])
        return result


# --------------------------------------------------------------------------- #
# 4. Reasoning Agent
# --------------------------------------------------------------------------- #
class ReasoningAgent(BaseAgent):
    """
    Role & Objective: Verify the initial sentiment by aggregating evidence,
    detecting contradictions, and finalizing the sentiment decision.
    """

    system_prompt = (
        "You are the Reasoning Agent in a multi-agent sentiment analysis "
        "framework. Evaluate the initial sentiment prediction using the "
        "original text, contextual information, external knowledge, "
        "aspect-level sentiment, and supporting evidence. Identify "
        "positive and negative evidence, examine potential "
        "contradictions, and determine whether the initial prediction is "
        "supported. Produce a final sentiment class and confidence value. "
        "The allowed classes are Positive, Negative, Neutral, and Mixed. "
        "Use only evidence supported by the input and supplied "
        "intermediate representations. Respond as JSON with exactly these "
        'keys: {"positive_evidence": [string], "negative_evidence": '
        '[string], "contradictions": [string], "final_label": string, '
        '"confidence": number}'
    )

    def run(
        self,
        review_text: str,
        context_output: Dict[str, Any],
        knowledge_output: Dict[str, Any],
        sentiment_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Input: X, O_C, O_K, O_S. Output: O_R (structured JSON)."""
        result = self._call(
            {
                "X": review_text,
                "O_C": context_output,
                "O_K": knowledge_output,
                "O_S": sentiment_output,
            }
        )
        result.setdefault("positive_evidence", [])
        result.setdefault("negative_evidence", [])
        result.setdefault("contradictions", [])
        result.setdefault("final_label", sentiment_output.get("initial_label", "Neutral"))
        result.setdefault("confidence", sentiment_output.get("confidence", 0.5))
        result["final_label"] = _coerce_label(result["final_label"])
        result["confidence"] = _clamp01(result["confidence"])
        return result


# --------------------------------------------------------------------------- #
# 5. Explainability Agent
# --------------------------------------------------------------------------- #
class ExplainabilityAgent(BaseAgent):
    """
    Role & Objective: Generate a concise, faithful, and human-readable
    explanation of the final sentiment decision using validated evidence.
    """

    system_prompt = (
        "You are the Explainability Agent in a multi-agent sentiment "
        "analysis framework. Generate a concise and human-readable "
        "explanation of the final sentiment decision using only the "
        "supplied text and validated evidence. Identify the principal "
        "sentiment-bearing aspects and explain how the positive and/or "
        "negative evidence supports the final class. Do not introduce new "
        "facts or alter the final sentiment decision. Respond as JSON "
        'with exactly these keys: {"final_label": string, "confidence": '
        'number, "explanation": string}'
    )

    def run(
        self,
        review_text: str,
        context_output: Dict[str, Any],
        knowledge_output: Dict[str, Any],
        sentiment_output: Dict[str, Any],
        reasoning_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Input: X, O_C, O_K, O_S, O_R. Output: O_E (structured JSON)."""
        result = self._call(
            {
                "X": review_text,
                "O_C": context_output,
                "O_K": knowledge_output,
                "O_S": sentiment_output,
                "O_R": reasoning_output,
            }
        )
        result.setdefault("final_label", reasoning_output.get("final_label", "Neutral"))
        result.setdefault("confidence", reasoning_output.get("confidence", 0.5))
        result.setdefault("explanation", "")
        # The Explainability Agent must not alter the Reasoning Agent's
        # decision -- enforce that invariant here rather than trusting the
        # LLM to always honor the instruction.
        result["final_label"] = reasoning_output.get("final_label", result["final_label"])
        result["confidence"] = reasoning_output.get("confidence", result["confidence"])
        return result


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _coerce_label(label: str) -> str:
    if not isinstance(label, str):
        return "Neutral"
    normalized = label.strip().capitalize()
    return normalized if normalized in ALLOWED_LABELS else "Neutral"


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))
