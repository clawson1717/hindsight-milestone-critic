"""
Milestone Evidence Collector — gather supporting evidence for a Milestone
from an agent trajectory.

Extracts text snippets, structured citations, and intermediate results using:
  - Keyword / phrase matching against the milestone description
  - Action-pattern matching against trajectory steps
  - Observation parsing (structured key-value and JSON)
  - Confidence scoring based on match quality and coverage
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.decompose import Milestone

# ---------------------------------------------------------------------------
# Action-verb patterns relevant to evidence extraction
# ---------------------------------------------------------------------------
_EVIDENCE_ACTION_PATTERNS = [
    re.compile(r"\banalyze[d]?\b", re.IGNORECASE),
    re.compile(r"\bretrieve[sd]?\b", re.IGNORECASE),
    re.compile(r"\bcompute[sd]?\b", re.IGNORECASE),
    re.compile(r"\bverify[sd]?\b", re.IGNORECASE),
    re.compile(r"\bsynthesize[sd]?\b", re.IGNORECASE),
    re.compile(r"\bevaluate[sd]?\b", re.IGNORECASE),
    re.compile(r"\bcompare[sd]?\b", re.IGNORECASE),
    re.compile(r"\bgenerate[sd]?\b", re.IGNORECASE),
    re.compile(r"\bselect[ed]?\b", re.IGNORECASE),
    re.compile(r"\bdecide[sd]?\b", re.IGNORECASE),
    re.compile(r"\bfetch[ed]?\b", re.IGNORECASE),
    re.compile(r"\bobtain[ed]?\b", re.IGNORECASE),
    re.compile(r"\bidentify[ed]?\b", re.IGNORECASE),
    re.compile(r"\bextract[ed]?\b", re.IGNORECASE),
    re.compile(r"\bbuild[ed]?\b", re.IGNORECASE),
    re.compile(r"\bcreate[ed]?\b", re.IGNORECASE),
    re.compile(r"\bcheck[ed]?\b", re.IGNORECASE),
    re.compile(r"\bmeasure[ed]?\b", re.IGNORECASE),
    re.compile(r"\bcalculate[ed]?\b", re.IGNORECASE),
    re.compile(r"\bquery[ed]?\b", re.IGNORECASE),
    re.compile(r"\bfind[ing]?\b", re.IGNORECASE),
    re.compile(r"\blocate[ed]?\b", re.IGNORECASE),
    re.compile(r"\bcollect[ed]?\b", re.IGNORECASE),
    re.compile(r"\binterpret[ed]?\b", re.IGNORECASE),
    re.compile(r"\bvalidate[ed]?\b", re.IGNORECASE),
    re.compile(r"\breport[ed]?\b", re.IGNORECASE),
    re.compile(r"\bsummarize[ed]?\b", re.IGNORECASE),
    re.compile(r"\bconclude[ed]?\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """
    Evidence gathered for a single milestone from a trajectory.

    Attributes
    ----------
    milestone_id : str
        UUID string of the milestone this evidence belongs to.
    text_snippets : List[str]
        Relevant text excerpts extracted from the trajectory.
    citations : List[Dict]
        Structured citations. Each dict contains:
        - ``source_action_idx``: zero-based index in the trajectory
        - ``text``: the quoted / extracted text
    intermediate_results : List[Any]
        Parsed structured data extracted from observations
        (e.g. JSON objects, key-value pairs, numeric results).
    confidence : float
        Evidence strength score in [0.0, 1.0].
        0.0 = no evidence found, 1.0 = strong, unambiguous evidence.
    """

    milestone_id: str
    text_snippets: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: List[Any] = field(default_factory=list)
    confidence: float = 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "milestone_id": self.milestone_id,
            "text_snippets": self.text_snippets,
            "citations": self.citations,
            "intermediate_results": self.intermediate_results,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        """Reconstruct an Evidence object from a dictionary."""
        return cls(
            milestone_id=data["milestone_id"],
            text_snippets=list(data.get("text_snippets", [])),
            citations=list(data.get("citations", [])),
            intermediate_results=list(data.get("intermediate_results", [])),
            confidence=float(data.get("confidence", 0.0)),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        snippet_count = len(self.text_snippets)
        citation_count = len(self.citations)
        result_count = len(self.intermediate_results)
        return (
            f"Evidence(milestone_id={self.milestone_id!r}, "
            f"snippets={snippet_count}, citations={citation_count}, "
            f"results={result_count}, confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# EvidenceCollector
# ---------------------------------------------------------------------------


class EvidenceCollector:
    """
    Gathers supporting evidence for a :class:`Milestone` from an agent
    trajectory.

    Parameters
    ----------
    min_confidence_threshold : float, optional
        Minimum confidence score for a piece of evidence to be retained.
        Defaults to 0.0 (keep everything).
    text_match_weight : float
        Weight for text-match evidence in confidence scoring.
        Defaults to 0.4.
    action_match_weight : float
        Weight for action-pattern evidence in confidence scoring.
        Defaults to 0.3.
    observation_match_weight : float
        Weight for observation-parsing evidence in confidence scoring.
        Defaults to 0.3.

    Examples
    --------
    >>> collector = EvidenceCollector()
    >>> milestones = decomposer.decompose("Analyze revenue and generate a report")
    >>> trajectory = [
    ...     {"action": "analyze revenue", "observation": "Revenue = $1.2M", "thought": "...", "timestamp": "2024-01-01T00:00:00Z"},
    ...     {"action": "generate PDF", "observation": "Report saved", "thought": "...", "timestamp": "2024-01-01T00:01:00Z"},
    ... ]
    >>> evidence_list = [collector.collect(m, trajectory) for m in milestones]
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.0,
        text_match_weight: float = 0.4,
        action_match_weight: float = 0.3,
        observation_match_weight: float = 0.3,
    ) -> None:
        if not (0.0 <= min_confidence_threshold <= 1.0):
            raise ValueError("min_confidence_threshold must be in [0.0, 1.0]")
        total = text_match_weight + action_match_weight + observation_match_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Weight sum ({total}) must equal 1.0; got "
                f"text={text_match_weight}, action={action_match_weight}, "
                f"observation={observation_match_weight}"
            )
        self._min_threshold = min_confidence_threshold
        self._text_weight = text_match_weight
        self._action_weight = action_match_weight
        self._obs_weight = observation_match_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        milestone: Milestone,
        trajectory: List[Dict[str, Any]],
    ) -> Evidence:
        """
        Collect evidence for a single milestone from a trajectory.

        Parameters
        ----------
        milestone : Milestone
            The milestone to gather evidence for.
        trajectory : List[Dict[str, Any]]
            List of trajectory steps. Each dict must contain at least
            ``action`` (str) and optionally ``observation`` (str),
            ``thought`` (str), and ``timestamp`` (str).

        Returns
        -------
        Evidence
            Structured evidence object with text snippets, citations,
            intermediate results, and a confidence score in [0.0, 1.0].
        """
        if not trajectory:
            return Evidence(
                milestone_id=str(milestone.id),
                confidence=0.0,
            )

        text_snippets: List[str] = []
        citations: List[Dict[str, Any]] = []
        intermediate_results: List[Any] = []

        # --- 1. Text / keyword matching ---------------------------------
        text_score, text_matches = self._score_text_match(milestone, trajectory)
        for match_str, action_idx in text_matches:
            if match_str not in text_snippets:
                text_snippets.append(match_str)
            citations.append({
                "source_action_idx": action_idx,
                "text": match_str,
            })

        # --- 2. Action-pattern matching ---------------------------------
        action_score, action_matches = self._score_action_pattern(
            milestone, trajectory
        )
        for match_str, action_idx in action_matches:
            if match_str not in text_snippets:
                text_snippets.append(match_str)
            citations.append({
                "source_action_idx": action_idx,
                "text": match_str,
            })

        # --- 3. Observation parsing -------------------------------------
        obs_score, parsed = self._parse_observations(milestone, trajectory)
        for item in parsed:
            intermediate_results.append(item)

        # --- 4. Confidence score ----------------------------------------
        raw_confidence = (
            text_score * self._text_weight
            + action_score * self._action_weight
            + obs_score * self._obs_weight
        )
        confidence = max(0.0, min(1.0, raw_confidence))

        # Deduplicate citations by (source_action_idx, text)
        seen: set[tuple[int, str]] = set()
        unique_citations: List[Dict[str, Any]] = []
        for c in citations:
            key = (c["source_action_idx"], c["text"])
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=text_snippets,
            citations=unique_citations,
            intermediate_results=intermediate_results,
            confidence=confidence,
        )

        # Apply minimum threshold
        if evidence.confidence < self._min_threshold:
            evidence.confidence = 0.0
            evidence.text_snippets = []
            evidence.citations = []
            evidence.intermediate_results = []

        return evidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text (word tokens ≥3 chars)."""
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        # Filter out common stop words
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "her", "was", "one", "our", "out", "day", "get",
            "has", "him", "his", "how", "its", "may", "new", "now",
            "old", "see", "two", "who", "boy", "did", "she", "use",
            "way", "will", "with", "from", "this", "that", "have",
            "more", "then", "into", "some", "would", "make", "like",
            "just", "over", "such", "take", "only", "come", "these",
            "could", "first", "after", "most", "also", "back",
        }
        return [t for t in tokens if t not in stop_words]

    def _score_text_match(
        self,
        milestone: Milestone,
        trajectory: List[Dict[str, Any]],
    ) -> tuple[float, List[tuple[str, int]]]:
        """
        Score text/keyword match between milestone description and
        trajectory entries.

        Returns
        -------
        score : float
            Normalised score in [0.0, 1.0].
        matches : List[tuple[str, int]]
            List of (matched_text, action_index) pairs.
        """
        desc_lower = milestone.description.lower()
        keywords = self._extract_keywords(milestone.description)

        matches: List[tuple[str, int]] = []
        matched_keywords: set[str] = set()

        for idx, step in enumerate(trajectory):
            step_text = " ".join(
                str(step.get(field, ""))
                for field in ("action", "observation", "thought")
            ).lower()

            # Phrase-level match: look for the full description or
            # significant fragments
            desc_words = desc_lower.split()
            for phrase_len in range(3, 0, -1):
                for i in range(len(desc_words) - phrase_len + 1):
                    phrase = " ".join(desc_words[i:i + phrase_len])
                    if len(phrase) >= 10 and phrase in step_text:
                        matches.append((phrase.strip(), idx))
                        matched_keywords.update(phrase.split())
                        break

            # Keyword-level match
            for kw in keywords:
                if len(kw) >= 4 and re.search(rf"\b{re.escape(kw)}\b", step_text):
                    if kw not in matched_keywords:
                        matches.append((kw, idx))
                        matched_keywords.add(kw)

        # Normalise score
        if not keywords:
            score = 0.0
        else:
            unique_matches = len(matched_keywords)
            score = min(1.0, unique_matches / max(1.0, len(keywords)))

        return score, matches

    def _score_action_pattern(
        self,
        milestone: Milestone,
        trajectory: List[Dict[str, Any]],
    ) -> tuple[float, List[tuple[str, int]]]:
        """
        Score how well trajectory actions match expected action patterns
        implied by the milestone.

        Returns
        -------
        score : float
            Normalised score in [0.0, 1.0].
        matches : List[tuple[str, action_idx]]
        """
        # Identify the action verb in the milestone description
        milestone_lower = milestone.description.lower()
        milestone_verb: Optional[str] = None
        for pattern in _EVIDENCE_ACTION_PATTERNS:
            m = pattern.search(milestone_lower)
            if m:
                milestone_verb = m.group(0).lower()
                break

        if not milestone_verb:
            return 0.0, []

        matches: List[tuple[str, int]] = []
        verb_found = False

        for idx, step in enumerate(trajectory):
            action_str = str(step.get("action", "")).lower()
            if milestone_verb in action_str:
                matches.append((action_str.strip(), idx))
                verb_found = True

        if verb_found:
            # One match is sufficient for a base score; coverage adds more
            score = min(1.0, len(matches) * 0.4)
        else:
            score = 0.0

        return score, matches

    def _parse_observations(
        self,
        milestone: Milestone,
        trajectory: List[Dict[str, Any]],
    ) -> tuple[float, List[Any]]:
        """
        Parse structured data from observations that are relevant to
        the milestone.

        Extracts:
          - JSON objects / arrays
          - Key-value pairs (e.g. "key: value" or "key = value")
          - Numeric results

        Returns
        -------
        score : float
            Score in [0.0, 1.0] based on richness of parsed data.
        parsed : List[Any]
            List of parsed values.
        """
        parsed: List[Any] = []
        milestone_lower = milestone.description.lower()
        milestone_keywords = set(self._extract_keywords(milestone.description))

        for step in trajectory:
            obs = str(step.get("observation", ""))

            # --- JSON extraction ---
            for json_match in self._extract_json(obs):
                # json_match is the parsed Python object (dict/list)
                parsed.append(json_match)
                # Serialise back to string for relevance check
                json_repr = json.dumps(json_match) if isinstance(json_match, (dict, list)) else str(json_match)
                if any(kw in json_repr.lower() for kw in milestone_keywords):
                    parsed.append(("relevance_tag", "keyword_match"))

            # --- Key-value pairs ---
            for kv in self._extract_key_values(obs):
                parsed.append(kv)
                if any(k.lower() in milestone_lower for k in kv.keys()):
                    parsed.append(("kv_relevance_tag", list(kv.keys())))

            # --- Numeric results ---
            for num_str, num_val in self._extract_numbers(obs):
                parsed.append(num_val)

        # Score based on variety and volume of parsed data
        if not parsed:
            score = 0.0
        else:
            # Cap score; more unique parsed items → higher score up to 1.0
            score = min(1.0, len(parsed) / 10.0)

        return score, parsed

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> List[Any]:
        """Attempt to extract JSON objects and arrays from text."""
        results: List[Any] = []
        # Find JSON objects {...} or arrays [...]
        for match in re.finditer(r"\{[^{}]*\}|\[[^\[\]]*\]", text):
            try:
                obj = json.loads(match.group(0))
                results.append(obj)
            except (json.JSONDecodeError, ValueError):
                pass
        return results

    @staticmethod
    def _extract_key_values(text: str) -> List[Dict[str, str]]:
        """Extract key-value pairs like 'key: value' or 'key = value'."""
        results: List[Dict[str, str]] = []
        # Match "key: value" or "key = value" patterns
        for match in re.finditer(
            r"(?<![a-zA-Z0-9])([a-zA-Z_][a-zA-Z0-9_\s]*?)\s*[=:]\s*([^\n,;]+)",
            text,
        ):
            key = match.group(1).strip()
            val = match.group(2).strip()
            if len(key) > 0 and len(val) > 0:
                results.append({key: val})
        return results

    @staticmethod
    def _extract_numbers(text: str) -> List[tuple[str, float]]:
        """Extract decimal/integer numbers from text."""
        results: List[tuple[str, float]] = []
        for match in re.finditer(r"(?<![a-zA-Z0-9])(-?\d+\.?\d*)\b", text):
            try:
                val = float(match.group(1))
                results.append((match.group(1), val))
            except ValueError:
                pass
        return results
