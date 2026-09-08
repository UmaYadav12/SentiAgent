"""
test_pipeline.py
-----------------
Unit tests that exercise the pipeline wiring without making real network
or API calls. A fake LLMClient returns deterministic canned JSON so the
tests validate data flow between agents, not model quality.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentiagent.pipeline import SentiAgentPipeline  # noqa: E402


class FakeLLMClient:
    """Drop-in replacement for LLMClient that returns scripted responses."""

    def __init__(self, *args, **kwargs):
        self.call_count = 0

    def complete_json(self, system_prompt: str, user_content: str):
        self.call_count += 1
        if "Context Agent" in system_prompt:
            return {
                "entities": ["phone"],
                "aspects": [
                    {"aspect": "camera", "evidence": "camera is excellent"},
                    {"aspect": "battery", "evidence": "battery drains quickly"},
                ],
                "context": "Product review about a phone.",
                "negation": [],
                "evidence": ["camera is excellent", "battery drains quickly"],
            }
        if "Knowledge Agent" in system_prompt:
            return {
                "concepts": ["battery life", "user satisfaction"],
                "relations": [
                    {"relation": "Causes", "source": "ConceptNet"}
                ],
                "relevance": "Poor battery life decreases user satisfaction.",
            }
        if "Sentiment Agent" in system_prompt:
            return {
                "aspect_sentiments": [
                    {"aspect": "camera", "sentiment": "Positive"},
                    {"aspect": "battery", "sentiment": "Negative"},
                ],
                "initial_label": "Mixed",
                "confidence": 0.93,
                "evidence": ["camera is excellent", "battery drains quickly"],
            }
        if "Reasoning Agent" in system_prompt:
            return {
                "positive_evidence": ["camera is excellent"],
                "negative_evidence": ["battery drains quickly"],
                "contradictions": [],
                "final_label": "Mixed",
                "confidence": 0.96,
            }
        if "Explainability Agent" in system_prompt:
            return {
                "final_label": "Mixed",
                "confidence": 0.96,
                "explanation": (
                    "The review appreciates camera quality but expresses "
                    "dissatisfaction with battery performance, so the "
                    "sentiment is classified as Mixed."
                ),
            }
        raise AssertionError(f"Unexpected system prompt: {system_prompt[:50]}")


def test_pipeline_end_to_end():
    with patch("sentiagent.pipeline.LLMClient", FakeLLMClient):
        pipeline = SentiAgentPipeline()
        result = pipeline.run(
            "The phone camera is excellent, but the battery drains quickly."
        )

    assert result.final_label == "Mixed"
    assert result.confidence == 0.96
    assert "camera" in result.explanation.lower() or "battery" in result.explanation.lower()
    assert result.context_output["aspects"][0]["aspect"] == "camera"
    assert result.sentiment_output["initial_label"] == "Mixed"
    assert result.reasoning_output["final_label"] == "Mixed"


def test_explainability_agent_cannot_override_final_label():
    """The Explainability Agent must not alter the Reasoning Agent's decision."""
    with patch("sentiagent.pipeline.LLMClient", FakeLLMClient):
        pipeline = SentiAgentPipeline()
        result = pipeline.run("Great sound, terrible battery life.")

    assert result.final_label == result.reasoning_output["final_label"]
    assert result.confidence == result.reasoning_output["confidence"]


if __name__ == "__main__":
    test_pipeline_end_to_end()
    test_explainability_agent_cannot_override_final_label()
    print("All tests passed.")
