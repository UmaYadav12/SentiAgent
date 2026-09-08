"""
llm_client.py
-------------
Thin wrapper around an LLM backend used by every agent in the SentiAgent
pipeline. Defaults to the Anthropic Messages API but can be swapped for any
provider (OpenAI, local model server, etc.) by implementing the same
`complete_json` interface.

The wrapper enforces JSON-only responses from the model, since every agent
in the framework communicates via structured JSON (see the paper's
Figures 1-2: each agent's "Output from Agent" block is a JSON schema).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


class LLMClient:
    """
    Provider-agnostic LLM client.

    Parameters
    ----------
    model : str
        Model identifier passed to the backend (e.g. "claude-sonnet-4-6").
    temperature : float
        Sampling temperature. Kept low by default for reproducibility,
        which matters for the reported accuracy figures.
    max_tokens : int
        Maximum tokens to generate per call.
    backend : str
        Which SDK to use. Currently supports "anthropic" (default) and
        "openai". Extend `_call_anthropic` / `_call_openai` or add your own
        `_call_<backend>` method to support additional providers.
    api_key : Optional[str]
        API key. If not provided, read from the standard environment
        variable for the chosen backend (ANTHROPIC_API_KEY / OPENAI_API_KEY).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        backend: str = "anthropic",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = backend
        self.api_key = api_key
        self._client = None  # lazily initialized

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete_json(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        """
        Send a system + user prompt to the LLM and parse a JSON object out
        of the response. Raises ValueError if no valid JSON can be parsed.
        """
        raw_text = self._dispatch(system_prompt, user_content)
        return self._extract_json(raw_text)

    # ------------------------------------------------------------------ #
    # Backend dispatch
    # ------------------------------------------------------------------ #
    def _dispatch(self, system_prompt: str, user_content: str) -> str:
        if self.backend == "anthropic":
            return self._call_anthropic(system_prompt, user_content)
        elif self.backend == "openai":
            return self._call_openai(system_prompt, user_content)
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def _call_anthropic(self, system_prompt: str, user_content: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for backend='anthropic'. "
                "Install it with: pip install anthropic"
            ) from exc

        if self._client is None:
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt
            + "\n\nRespond with a single valid JSON object only. "
              "Do not include markdown code fences, explanations, or any "
              "text outside the JSON object.",
            messages=[{"role": "user", "content": user_content}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    def _call_openai(self, system_prompt: str, user_content: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for backend='openai'. "
                "Install it with: pip install openai"
            ) from exc

        if self._client is None:
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key)

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract the first valid JSON object from a raw LLM response."""
        text = text.strip()
        # Strip accidental markdown fences.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: greedily find the outermost {...} block.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON from LLM output: {exc}\nRaw: {text}")
        raise ValueError(f"No JSON object found in LLM output.\nRaw: {text}")
