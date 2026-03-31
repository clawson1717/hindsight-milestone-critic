"""
Hindsight-Aware Policy — shapes policy decisions away from previously-failed modes.

This module implements Step 6 of the HiMCA pipeline: given a task and its
milestones, the HindsightAwarePolicy enriches the decision context with
relevant failed-experience entries from the HindsightStore so the policy
can avoid repeating previously-failed approaches.

The policy acts as a wrapper around a "base policy" (any callable that
produces an action for a task). It intercepts the task, enriches the prompt
with hindsight context, and returns a structured PolicyDecision that
includes the hindsight entries used to shape the decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from src.decompose import Milestone
from src.hindsight import HindsightEntry, HindsightStore

# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    """
    The output of a policy decision — an action plus metadata.

    Attributes
    ----------
    action : str
        The action string selected by the policy.
    reasoning : str
        Free-text explanation of why this action was chosen.
    confidence : float
        Confidence score in [0.0, 1.0] for this decision.
    hindsight_entries_used : List[HindsightEntry]
        The hindsight entries that were used to shape this decision.
    shaped_context : str
        The enriched context/prompt that was passed to the base policy
        (includes hindsight context).
    """

    action: str
    reasoning: str = ""
    confidence: float = 1.0
    hindsight_entries_used: List[HindsightEntry] = field(default_factory=list)
    shaped_context: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "hindsight_entries_used": [
                entry.to_dict() for entry in self.hindsight_entries_used
            ],
            "shaped_context": self.shaped_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyDecision:
        """Reconstruct from a dictionary."""
        entries = [
            HindsightEntry.from_dict(e) for e in data.get("hindsight_entries_used", [])
        ]
        return cls(
            action=str(data["action"]),
            reasoning=str(data.get("reasoning", "")),
            confidence=float(data.get("confidence", 1.0)),
            hindsight_entries_used=entries,
            shaped_context=str(data.get("shaped_context", "")),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PolicyDecision(action={self.action[:40]!r}, "
            f"confidence={self.confidence:.2f}, "
            f"hindsight_entries={len(self.hindsight_entries_used)})"
        )


# ---------------------------------------------------------------------------
# Base policy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BasePolicy(Protocol):
    """
    Protocol for a base policy callable.

    A base policy takes a task string (or shaped context string) and
    returns an action string. Any callable matching this signature can
    be used as the base policy for HindsightAwarePolicy.
    """

    def __call__(self, task_context: str, **kwargs: Any) -> str:
        ...


# ---------------------------------------------------------------------------
# Relevance scoring utilities
# ---------------------------------------------------------------------------

# Stop words reused from hindsight.py
_POLICY_STOP_WORDS: set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "her", "was", "one", "our", "out", "day", "get",
    "has", "him", "his", "how", "its", "may", "new", "now",
    "old", "see", "two", "who", "boy", "did", "she", "use",
    "way", "will", "with", "from", "this", "that", "have",
    "more", "then", "into", "some", "would", "make", "like",
    "just", "over", "such", "take", "only", "come", "these",
    "could", "first", "after", "most", "also", "back",
    "than", "them", "same", "well", "about", "being", "very",
    "your", "what", "when", "where", "which", "while", "why",
    "how", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so",
    "than", "too", "very", "should", "now", "because",
    "while", "either", "neither", "nor", "yet", "here",
    "there", "up", "down", "in", "out", "on", "off", "again",
    "further", "once",
}


def _extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """Extract significant keywords from text."""
    tokens = re.findall(r"\b[a-zA-Z]{" + str(min_len) + r",}\b", text.lower())
    return [t for t in tokens if t not in _POLICY_STOP_WORDS]


def _compute_keyword_overlap(a: List[str], b: List[str]) -> float:
    """
    Compute Jaccard-like keyword overlap between two keyword lists.

    Returns a score in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_relevance_score(
    task: str,
    entry: HindsightEntry,
) -> float:
    """
    Compute a relevance score between a task and a hindsight entry.

    Uses a combination of:
    1. Task vs. entry task keyword overlap
    2. Task vs. failed milestone description overlap

    Returns a score in [0.0, 1.0].
    """
    task_keywords = _extract_keywords(task)
    entry_task_keywords = _extract_keywords(entry.original_task)
    entry_milestone_keywords = _extract_keywords(entry.failed_milestone.description)

    # Task ↔ original task keyword overlap
    task_overlap = _compute_keyword_overlap(task_keywords, entry_task_keywords)

    # Task ↔ failed milestone description overlap
    milestone_overlap = _compute_keyword_overlap(task_keywords, entry_milestone_keywords)

    # Combine with equal weight
    return 0.5 * task_overlap + 0.5 * milestone_overlap


# ---------------------------------------------------------------------------
# HindsightAwarePolicy
# ---------------------------------------------------------------------------


class HindsightAwarePolicy:
    """
    A policy wrapper that shapes decisions away from previously-failed modes.

    This policy wraps a base policy callable and enriches its context with
    relevant hindsight entries from a HindsightStore before calling the base
    policy. The enriched context warns the policy about similar tasks that
    have previously failed, allowing it to explore alternative approaches.

    Parameters
    ----------
    base_policy : Callable[[str], str]
        The underlying policy that produces an action given a task context.
        Must accept a string (task or shaped context) and return an action string.
    hindsight_store : HindsightStore
        The hindsight experience store to query for relevant failures.
    top_k : int, optional
        Maximum number of hindsight entries to include in the shaped context.
        Defaults to 5.
    relevance_threshold : float, optional
        Minimum relevance score (0.0–1.0) for a hindsight entry to be included.
        Defaults to 0.0 (include all retrieved entries up to top_k).
    shaping_method : str, optional
        How to incorporate hindsight entries into the context.
        - ``"append"``: Append a HINDSIGHT_SECTION to the task string.
        - ``"prepend"``: Prepend the HINDSIGHT_SECTION to the task string.
        Defaults to ``"append"``.

    Examples
    --------
    >>> from src.policy import HindsightAwarePolicy, PolicyDecision
    >>> from src.hindsight import HindsightStore
    >>> from src.decompose import MilestoneDecomposer
    >>> from src.critic import MilestoneCritic
    >>> from src.evidence import EvidenceCollector

    >>> # Set up components
    >>> store = HindsightStore()
    >>> decomposer = MilestoneDecomposer()
    >>> collector = EvidenceCollector()
    >>> critic = MilestoneCritic()

    >>> # Simple base policy that just returns a default action
    >>> def base_policy(task: str) -> str:
    ...     if "revenue" in task.lower():
    ...         return "analyze_revenue"
    ...     return "default_action"

    >>> # Wrap with hindsight awareness
    >>> policy = HindsightAwarePolicy(base_policy, store, top_k=3)

    >>> # Get a shaped decision
    >>> milestones = decomposer.decompose("Analyze Q3 revenue and generate a report")
    >>> decision = policy.get_action("Analyze Q3 revenue and generate a report", milestones)
    >>> print(decision.action)
    analyze_revenue
    >>> print(f"Hindsight entries used: {len(decision.hindsight_entries_used)}")
    Hindsight entries used: 0
    """

    # Section header used to separate hindsight context from the original task
    HINDSIGHT_SECTION_HEADER = "\n\n## HINDSIGHT CONTEXT (Previous Failures)\n"
    HINDSIGHT_SECTION_FOOTER = "\n## END HINDSIGHT CONTEXT\n"

    def __init__(
        self,
        base_policy: Callable[[str], str],
        hindsight_store: HindsightStore,
        top_k: int = 5,
        relevance_threshold: float = 0.0,
        shaping_method: str = "append",
    ) -> None:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not (0.0 <= relevance_threshold <= 1.0):
            raise ValueError(
                f"relevance_threshold must be in [0.0, 1.0], got {relevance_threshold}"
            )
        if shaping_method not in ("append", "prepend"):
            raise ValueError(
                f"shaping_method must be 'append' or 'prepend', got {shaping_method!r}"
            )

        self._base_policy = base_policy
        self._store = hindsight_store
        self._top_k = top_k
        self._relevance_threshold = relevance_threshold
        self._shaping_method = shaping_method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(
        self,
        task: str,
        milestones: Optional[List[Milestone]] = None,
    ) -> PolicyDecision:
        """
        Get a policy decision shaped by relevant hindsight entries.

        Parameters
        ----------
        task : str
            The original task description.
        milestones : List[Milestone], optional
            The decomposed milestones for this task. If provided, used to
            improve hindsight retrieval relevance.

        Returns
        -------
        PolicyDecision
            A structured decision containing the action, reasoning,
            confidence, hindsight entries used, and the shaped context.
        """
        # Step 1: Retrieve relevant hindsight entries from the store
        hindsight_entries = self._retrieve_relevant_entries(task, milestones)

        # Step 2: Shape the context/prompt with hindsight entries
        shaped_context = self.shape_prompt(task, hindsight_entries)

        # Step 3: Call the base policy with the shaped context
        try:
            action = self._base_policy(shaped_context)
        except Exception as e:
            # If the base policy fails, fall back to a safe default
            action = f"fallback_action_due_to_error: {e}"

        # Step 4: Compute confidence based on hindsight relevance
        confidence = self._compute_decision_confidence(hindsight_entries, action)

        # Step 5: Build reasoning string
        reasoning = self._build_reasoning(task, hindsight_entries, action)

        return PolicyDecision(
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            hindsight_entries_used=hindsight_entries,
            shaped_context=shaped_context,
        )

    def shape_prompt(
        self,
        task: str,
        hindsight_entries: List[HindsightEntry],
    ) -> str:
        """
        Enrich a task prompt with hindsight context.

        Builds a shaped context string by appending (or prepending) a
        HINDSIGHT_SECTION that describes relevant previous failures.
        Each entry contributes its hindsight label and a brief description
        of what went wrong, allowing the policy to avoid those approaches.

        Parameters
        ----------
        task : str
            The original task description.
        hindsight_entries : List[HindsightEntry]
            The relevant hindsight entries to incorporate.

        Returns
        -------
        str
            The enriched task context with hindsight context appended
            (or prepended) as a separate section.
        """
        if not hindsight_entries:
            return task

        hindsight_text = self._build_hindsight_section(hindsight_entries)

        if self._shaping_method == "prepend":
            return hindsight_text + "\n\n" + task
        else:
            return task + hindsight_text

    def get_relevant_hindsight(
        self,
        task: str,
        milestones: Optional[List[Milestone]] = None,
        top_k: Optional[int] = None,
    ) -> List[HindsightEntry]:
        """
        Retrieve relevant hindsight entries without making a policy decision.

        This is a convenience method for inspecting what hindsight entries
        would be used for a given task, without calling the base policy.

        Parameters
        ----------
        task : str
            The task description.
        milestones : List[Milestone], optional
            The milestones for the task.
        top_k : int, optional
            Override the default top_k for this call.

        Returns
        -------
        List[HindsightEntry]
            Relevant hindsight entries sorted by relevance score.
        """
        return self._retrieve_relevant_entries(
            task, milestones, top_k=top_k
        )

    def relevance_scores(
        self,
        task: str,
        hindsight_entries: List[HindsightEntry],
    ) -> List[float]:
        """
        Compute relevance scores for a list of hindsight entries against a task.

        Parameters
        ----------
        task : str
            The task description.
        hindsight_entries : List[HindsightEntry]
            The entries to score.

        Returns
        -------
        List[float]
            One relevance score per entry, in [0.0, 1.0].
        """
        return [_compute_relevance_score(task, entry) for entry in hindsight_entries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_relevant_entries(
        self,
        task: str,
        milestones: Optional[List[Milestone]] = None,
        top_k: Optional[int] = None,
    ) -> List[HindsightEntry]:
        """
        Retrieve relevant hindsight entries from the store.

        First uses HindsightStore.retrieve() to get candidate entries,
        then re-ranks by the local relevance scorer and filters by
        relevance_threshold.
        """
        k = top_k if top_k is not None else self._top_k

        if len(self._store) == 0:
            return []

        # Use the store's built-in retrieval with a generous top_k
        candidates = self._store.retrieve(
            task,
            top_k=k * 2 if k > 0 else 10,  # Fetch extra for re-ranking
            similarity_threshold=0.0,
        )

        if not candidates:
            return []

        # Re-rank by local relevance scorer
        scored: List[tuple[float, HindsightEntry]] = []
        for entry in candidates:
            score = _compute_relevance_score(task, entry)
            if score >= self._relevance_threshold:
                scored.append((score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top_k
        selected = [entry for _, entry in scored[:k] if k > 0]

        return selected

    def _build_hindsight_section(
        self,
        hindsight_entries: List[HindsightEntry],
    ) -> str:
        """
        Build a formatted HINDSIGHT_SECTION string from a list of entries.

        Format::

            ## HINDSIGHT CONTEXT (Previous Failures)

            ### Previous Failure 1
            Label: What if [milestone] was the actual goal?
            What happened: <verdict and brief critic reasoning>
            How to avoid: <derived guidance based on the failure>

            ### Previous Failure 2
            ...

            ## END HINDSIGHT CONTEXT
        """
        lines = [self.HINDSIGHT_SECTION_HEADER]

        for i, entry in enumerate(hindsight_entries, start=1):
            lines.append(f"### Previous Failure {i}")
            lines.append(f"Task: {entry.original_task}")
            lines.append(f"Failed milestone: {entry.failed_milestone.description}")

            # Verdict context
            verdict = entry.critic_result.verdict.value.upper()
            confidence_str = f"{entry.critic_result.confidence:.2f}"
            lines.append(
                f"Verdict: {verdict} (confidence={confidence_str})"
            )

            # Critic reasoning (abbreviated)
            reasoning = entry.critic_result.reasoning
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            lines.append(f"Critic reasoning: {reasoning}")

            # Hindsight label
            lines.append(f"Reflection: {entry.hindsight_label}")

            # Derived avoidance guidance
            avoidance = self._derive_avoidance_guidance(entry)
            lines.append(f"How to avoid this failure: {avoidance}")

            lines.append("")  # Blank line between entries

        lines.append(self.HINDSIGHT_SECTION_FOOTER.rstrip())
        return "\n".join(lines)

    def _derive_avoidance_guidance(self, entry: HindsightEntry) -> str:
        """
        Derive a short guidance string from a hindsight entry.

        Generates actionable advice based on the failed milestone's
        description and the critic's reasoning.
        """
        parts: List[str] = []

        # Heuristic: extract key failure modes from reasoning
        reasoning_lower = entry.critic_result.reasoning.lower()

        if "insufficient" in reasoning_lower or "incomplete" in reasoning_lower:
            parts.append("Ensure sufficient evidence is gathered before evaluating.")
        if "boundary violation" in reasoning_lower or "constraint" in reasoning_lower:
            parts.append("Double-check constraint compliance before proceeding.")
        if "ambiguous" in reasoning_lower:
            parts.append("Seek clearer evidence or break down the milestone further.")
        if "contradict" in reasoning_lower or "inconsistent" in reasoning_lower:
            parts.append("Verify internal consistency of intermediate results.")
        if "high-severity" in reasoning_lower:
            # Extract what the high-severity violation was
            for v in entry.critic_result.violations:
                if v.severity == "high":
                    parts.append(f"Prioritise: {v.description[:100]}")
                    break

        if not parts:
            # Generic fallback guidance
            milestone_keywords = _extract_keywords(entry.failed_milestone.description)
            if milestone_keywords:
                parts.append(
                    f"When working on similar tasks, pay special attention to "
                    f"{', '.join(milestone_keywords[:3])}."
                )
            else:
                parts.append(
                    "Review this failure mode carefully before attempting "
                    "a similar task."
                )

        return " ".join(parts)

    def _compute_decision_confidence(
        self,
        hindsight_entries: List[HindsightEntry],
        action: str,
    ) -> float:
        """
        Compute a confidence score for the policy decision.

        Based on:
        1. Number of relevant hindsight entries found
        2. Average critic confidence of those entries
        3. Whether the action is a fallback (indicates uncertainty)
        """
        if "fallback_action_due_to_error" in action:
            return 0.0

        if not hindsight_entries:
            # No hindsight entries found — neutral confidence
            return 0.5

        # Average critic confidence of retrieved entries
        avg_critic_confidence = sum(
            e.critic_result.confidence for e in hindsight_entries
        ) / len(hindsight_entries)

        # Boost confidence when we have relevant hindsight
        # (we're more informed, but also more constrained)
        base = 0.5
        entries_bonus = min(0.2, len(hindsight_entries) * 0.03)
        relevance_bonus = avg_critic_confidence * 0.2

        return min(1.0, base + entries_bonus + relevance_bonus)

    def _build_reasoning(
        self,
        task: str,
        hindsight_entries: List[HindsightEntry],
        action: str,
    ) -> str:
        """Build a human-readable reasoning string for the decision."""
        if not hindsight_entries:
            return (
                f"No relevant hindsight entries found in store. "
                f"Proceeding with base policy decision: {action!r}"
            )

        verdict_counts: Dict[str, int] = {}
        for entry in hindsight_entries:
            key = entry.critic_result.verdict.value
            verdict_counts[key] = verdict_counts.get(key, 0) + 1

        parts = [
            f"Found {len(hindsight_entries)} relevant past failure(s) "
            f"(verdicts: {verdict_counts}). "
            f"Shaped policy away from previously-failed modes. "
            f"Selected action: {action!r}."
        ]

        if "fallback_action_due_to_error" not in action:
            parts.append(
                f"Context enriched with {len(hindsight_entries)} hindsight entries "
                f"before passing to base policy."
            )
        else:
            parts.append("Base policy error encountered; used fallback action.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HindsightAwarePolicy(top_k={self._top_k}, "
            f"relevance_threshold={self._relevance_threshold}, "
            f"shaping_method={self._shaping_method!r}, "
            f"store_size={len(self._store)})"
        )
