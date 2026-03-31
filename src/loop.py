"""
HiMCA Integration Loop — HiMCALoop orchestrator.

Orchestrates the full HiMCA pipeline:
    Task → decompose → execute/verify milestones →
    critic evaluates each → store failures as hindsight →
    next attempt informed by hindsight → repeat until all milestones
    pass or max_attempts reached.

This module implements the top-level orchestration that ties together
the MilestoneDecomposer, EvidenceCollector, MilestoneCritic, and
HindsightStore into a single run-loop with early termination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.critic import MilestoneCritic, MilestoneCriticResult, MilestoneVerdict
from src.decompose import Milestone, MilestoneDecomposer
from src.evidence import Evidence, EvidenceCollector
from src.hindsight import HindsightEntry, HindsightStore

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LoopResult:
    """
    The outcome of a complete HiMCA loop run.

    Attributes
    ----------
    success : bool
        True if all milestones passed (at least one attempt ended with
        every milestone receiving a PASS verdict).
    attempts : int
        Number of attempts made before the loop terminated
        (1 to max_attempts, inclusive).
    milestones_verified : List[Milestone]
        The list of milestones that were decomposed from the task.
    milestone_verdicts : List[List[MilestoneCriticResult]]
        One list per attempt, containing the critic results for each milestone.
        Useful for post-hoc analysis of how verdicts evolved across attempts.
    hindsight_entries : List[HindsightEntry]
        All hindsight entries stored across all attempts.
        One entry per failed milestone per attempt.
    final_verdict : str
        Human-readable summary of the final outcome, e.g.
        "All 3 milestones PASSED on attempt 1" or
        "2 of 3 milestones still FAILING after 3 attempts".
    """

    success: bool
    attempts: int
    milestones_verified: List[Milestone] = field(default_factory=list)
    milestone_verdicts: List[List[MilestoneCriticResult]] = field(
        default_factory=list
    )
    hindsight_entries: List[HindsightEntry] = field(default_factory=list)
    final_verdict: str = ""


# ---------------------------------------------------------------------------
# Evidence provider type alias
# ---------------------------------------------------------------------------

# Signature: (milestones, attempt_number, hindsight_entries) -> List[Evidence]
# - milestones: the milestones to collect evidence for
# - attempt_number: 1-based attempt index (for logging / strategy)
# - hindsight_entries: all hindsight entries stored so far (to inform collection)
# - Returns: one Evidence object per milestone, aligned by index
EvidenceProvider = Callable[
    [List[Milestone], int, List[HindsightEntry]],
    List[Evidence],
]


def _default_evidence_provider(
    milestones: List[Milestone],
    attempt: int,
    hindsight_entries: List[HindsightEntry],
) -> List[Evidence]:
    """
   Default no-op evidence provider — returns empty evidence for every milestone.

    Override by passing your own ``evidence_provider`` to ``HiMCALoop.run``.
    """
    return [
        Evidence(
            milestone_id=str(m.id),
            text_snippets=[],
            citations=[],
            intermediate_results=[],
            confidence=0.0,
        )
        for m in milestones
    ]


# ---------------------------------------------------------------------------
# HiMCALoop
# ---------------------------------------------------------------------------


class HiMCALoop:
    """
    Orchestrates the HiMCA pipeline over multiple attempts.

    The loop implements the following flow:

    1. **Decompose** the task into ordered milestones.
    2. **Attempt loop** (1 → max_attempts):
       a. Collect evidence for each milestone (via ``evidence_provider``).
       b. Run the milestone critic on each (milestone + evidence).
       c. Store every FAIL / UNCERTAIN milestone in the hindsight store.
       d. If all milestones pass → early exit (success).
    3. Return a :class:`LoopResult` with the full outcome.

    Parameters
    ----------
    decomposer : MilestoneDecomposer, optional
        The decomposer to use. A default one is created if not supplied.
    evidence_collector : EvidenceCollector, optional
        The evidence collector to use. A default one is created if not supplied.
    critic : MilestoneCritic, optional
        The milestone critic to use. A default one is created if not supplied.
    hindsight_store : HindsightStore, optional
        The hindsight store for failed milestones.
        A new empty in-memory store is created if not supplied.
    evidence_provider : EvidenceProvider, optional
        Callback that receives the milestone list and attempt number and
        returns a list of :class:`Evidence` objects (one per milestone,
        aligned by index). If not supplied, a default no-op provider that
        returns empty evidence is used — override this for real use.
    max_attempts : int, optional
        Default maximum number of attempts (can be overridden per ``run`` call).
        Defaults to 3.

    Examples
    --------
    >>> from src.loop import HiMCALoop, LoopResult

    Basic usage with a custom evidence provider:

    >>> def my_evidence_provider(milestones, attempt, hindsight):
    ...     # In a real agent this would consult the actual agent trajectory.
    ...     # Here we return simulated evidence keyed on milestone description.
    ...     return [
    ...         Evidence(
    ...             milestone_id=str(m.id),
    ...             text_snippets=[f"Evidence for: {m.description}"],
    ...             citations=[],
    ...             intermediate_results=[],
    ...             confidence=0.9,
    ...         )
    ...         for m in milestones
    ...     ]

    >>> loop = HiMCALoop(
    ...     evidence_provider=my_evidence_provider,
    ...     max_attempts=3,
    ... )
    >>> result = loop.run("Analyze revenue and generate a report")
    >>> print(f"Success: {result.success}, Attempts: {result.attempts}")
    Success: ...

    Accessing per-attempt verdicts for analysis:

    >>> for attempt_idx, verdicts in enumerate(result.milestone_verdicts, 1):
    ...     for milestone, verdict in zip(result.milestones_verified, verdicts):
    ...         print(f"  Attempt {attempt_idx} | {milestone.description[:40]}: {verdict.verdict.value}")
    """

    def __init__(
        self,
        decomposer: Optional[MilestoneDecomposer] = None,
        evidence_collector: Optional[EvidenceCollector] = None,
        critic: Optional[MilestoneCritic] = None,
        hindsight_store: Optional[HindsightStore] = None,
        evidence_provider: Optional[EvidenceProvider] = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        self._decomposer = decomposer or MilestoneDecomposer()
        self._collector = evidence_collector or EvidenceCollector()
        self._critic = critic or MilestoneCritic()
        self._store = hindsight_store or HindsightStore()
        self._evidence_provider = evidence_provider or _default_evidence_provider
        self._default_max_attempts = max_attempts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        max_attempts: Optional[int] = None,
        evidence_provider: Optional[EvidenceProvider] = None,
    ) -> LoopResult:
        """
        Run the HiMCA loop for a single task.

        Parameters
        ----------
        task : str
            The natural-language task to decompose and verify.
        max_attempts : int, optional
            Override the default max_attempts for this run.
            Uses the constructor default if not supplied.
        evidence_provider : EvidenceProvider, optional
            Override the evidence provider for this run only.

        Returns
        -------
        LoopResult
            The complete outcome of the run, including success flag,
            number of attempts, all milestones and their verdicts,
            hindsight entries stored, and a human-readable final verdict.

        Raises
        ------
        ValueError
            If ``task`` is empty or ``max_attempts`` is < 1.
        """
        if not task or not task.strip():
            raise ValueError("Task description must not be empty.")

        max_att = max_attempts if max_attempts is not None else self._default_max_attempts
        if max_att < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_att}")

        evidence_fn = evidence_provider or self._evidence_provider

        # Step 1: Decompose task into milestones
        milestones = self._decomposer.decompose(task)

        if not milestones:
            return LoopResult(
                success=False,
                attempts=0,
                milestones_verified=[],
                milestone_verdicts=[],
                hindsight_entries=[],
                final_verdict="Task decomposed into no milestones — aborting.",
            )

        all_verdicts: List[List[MilestoneCriticResult]] = []
        all_hindsight_entries: List[HindsightEntry] = []
        all_trajectories: List[List[Dict[str, Any]]] = []

        # Step 2: Attempt loop
        for attempt in range(1, max_att + 1):
            # (a) Collect evidence for each milestone
            evidence_list = evidence_fn(milestones, attempt, all_hindsight_entries)

            # (b) Run critic on each milestone
            critic_results = self._critic.critique(milestones, evidence_list)
            all_verdicts.append(critic_results)

            # (c) Store failed / uncertain milestones as hindsight entries
            # We construct a minimal trajectory from the evidence (text snippets)
            trajectories_for_store = self._evidence_to_trajectory(evidence_list)
            all_trajectories.append(trajectories_for_store)

            stored_ids = self._store.add(
                task=task,
                milestones=milestones,
                evidence_list=evidence_list,
                critic_results=critic_results,
                trajectory=trajectories_for_store,
            )

            # Retrieve the entries we just stored (same order, same count)
            # The store's add() returns entry IDs; we fetch them back to
            # include in the result.
            if stored_ids:
                retrieved = [
                    entry
                    for entry in self._store._entries  # type: ignore[attr-defined]
                    if entry.entry_id in set(stored_ids)
                ]
                all_hindsight_entries.extend(retrieved)

            # (d) Early exit if all milestones pass
            if self._all_pass(critic_results):
                return LoopResult(
                    success=True,
                    attempts=attempt,
                    milestones_verified=milestones,
                    milestone_verdicts=all_verdicts,
                    hindsight_entries=all_hindsight_entries,
                    final_verdict=self._build_final_verdict(
                        milestones, all_verdicts, attempt, success=True
                    ),
                )

        # Exhausted all attempts without full success
        return LoopResult(
            success=False,
            attempts=max_att,
            milestones_verified=milestones,
            milestone_verdicts=all_verdicts,
            hindsight_entries=all_hindsight_entries,
            final_verdict=self._build_final_verdict(
                milestones, all_verdicts, max_att, success=False
            ),
        )

    @property
    def hindsight_store(self) -> HindsightStore:
        """The hindsight store used by this loop."""
        return self._store

    @property
    def decomposer(self) -> MilestoneDecomposer:
        """The decomposer used by this loop."""
        return self._decomposer

    @property
    def critic(self) -> MilestoneCritic:
        """The critic used by this loop."""
        return self._critic

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_pass(results: List[MilestoneCriticResult]) -> bool:
        """Return True iff every result has a PASS verdict."""
        return all(r.verdict == MilestoneVerdict.PASS for r in results)

    @staticmethod
    def _evidence_to_trajectory(
        evidence_list: List[Evidence],
    ) -> List[dict[str, Any]]:
        """
        Convert a list of Evidence objects into a minimal trajectory format.

        Each trajectory step is a dict with ``action``, ``observation``,
        and ``thought`` fields derived from the evidence snippets.
        This is used when storing hindsight entries so they retain the
        evidence context.
        """
        trajectory: List[Dict[str, Any]] = []
        for ev in evidence_list:
            for snippet in ev.text_snippets:
                trajectory.append({
                    "action": snippet,
                    "observation": snippet,
                    "thought": "",
                    "timestamp": "",
                })
            for result in ev.intermediate_results:
                trajectory.append({
                    "action": str(result),
                    "observation": str(result),
                    "thought": "",
                    "timestamp": "",
                })
        return trajectory

    @staticmethod
    def _build_final_verdict(
        milestones: List[Milestone],
        all_verdicts: List[List[MilestoneCriticResult]],
        attempts: int,
        success: bool,
    ) -> str:
        """Build a human-readable final verdict string."""
        n = len(milestones)

        if success:
            return (
                f"All {n} milestone(s) PASSED on attempt {attempts}. "
                f"Task completed successfully."
            )

        # Find the last round of verdicts
        final_verdicts = all_verdicts[-1] if all_verdicts else []
        fail_count = sum(
            1 for v in final_verdicts if v.verdict == MilestoneVerdict.FAIL
        )
        uncertain_count = sum(
            1 for v in final_verdicts if v.verdict == MilestoneVerdict.UNCERTAIN
        )

        parts = []
        if fail_count:
            parts.append(f"{fail_count} FAIL")
        if uncertain_count:
            parts.append(f"{uncertain_count} UNCERTAIN")

        verdict_str = ", ".join(parts) if parts else "all UNCERTAIN"

        return (
            f"{verdict_str} after {attempts} attempt(s). "
            f"{n} milestone(s) decomposed; not all verified. "
            f"Consider reviewing hindsight entries for failure patterns."
        )
