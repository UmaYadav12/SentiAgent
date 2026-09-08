"""
knowledge_sources.py
---------------------
External commonsense-knowledge retrieval used by the Knowledge Agent
(Figure 1, Agent 2: "Enrich the contextual representation using external
commonsense knowledge that is relevant to the identified aspects").

This module queries the public ConceptNet API for commonsense relations
around each extracted aspect. It is intentionally lightweight and has no
hard dependency on COMET; a COMET-based retriever can be plugged in via
the same `retrieve(aspects)` interface if a local COMET checkpoint is
available (see `CometKnowledgeRetriever` stub below).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import requests

CONCEPTNET_API = "https://api.conceptnet.io/c/en/{term}"


class ConceptNetRetriever:
    """Retrieves commonsense edges from ConceptNet for a list of aspect terms."""

    def __init__(self, limit_per_aspect: int = 5, timeout: float = 5.0):
        self.limit_per_aspect = limit_per_aspect
        self.timeout = timeout

    def retrieve(self, aspects: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Parameters
        ----------
        aspects : list of aspect strings (e.g. ["camera", "battery"]).

        Returns
        -------
        dict mapping each aspect -> list of {relation, start, end} edges.
        """
        knowledge: Dict[str, List[Dict[str, str]]] = {}
        for aspect in aspects:
            term = aspect.strip().lower().replace(" ", "_")
            if not term:
                continue
            try:
                resp = requests.get(
                    CONCEPTNET_API.format(term=term), timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException:
                knowledge[aspect] = []
                continue

            edges = []
            for edge in data.get("edges", [])[: self.limit_per_aspect]:
                rel = edge.get("rel", {}).get("label", "RelatedTo")
                start = edge.get("start", {}).get("label", "")
                end = edge.get("end", {}).get("label", "")
                if start and end:
                    edges.append({"relation": rel, "start": start, "end": end})
            knowledge[aspect] = edges
        return knowledge


class CometKnowledgeRetriever:
    """
    Optional COMET-based retriever stub.

    If you have a local COMET (e.g. COMET-ATOMIC2020) checkpoint, load it
    here and implement `retrieve` with the same signature as
    `ConceptNetRetriever.retrieve` so it can be swapped into the
    KnowledgeAgent without any other code changes.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            raise NotImplementedError(
                "Load your COMET checkpoint here, e.g. via the "
                "'comet-atomic-2020-bart' package, then implement retrieve()."
            )

    def retrieve(self, aspects: List[str]) -> Dict[str, List[Dict[str, str]]]:
        raise NotImplementedError("Implement COMET inference for each aspect.")
