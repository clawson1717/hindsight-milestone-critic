"""
Tests for src.loop — HiMCALoop orchestrator and LoopResult dataclass.
"""

from __future__ import annotations

import uuid

import pytest

from src.critic import (
    BoundaryViolation,
    MilestoneCritic,
    MilestoneCriticResult,
    MilestoneVerdict,
)
from src.decompose import Milestone, MilestoneDecomposer
from src.evidence import Evidence, EvidenceCollector
from src.hindsight import HindsightEntry, HindsightStore
from src.loop import (
    HiMCALoop,
    LoopResult,
    _default_evidence_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def decomposer() -> MilestoneDecomposer:
    return MilestoneDecomposer()


@pytest.fixture
def evidence_collector() -> EvidenceCollector:
    return EvidenceCollector()


@pytest.fixture
def critic() -> MilestoneCritic:
    return MilestoneCritic()


@pytest.fixture
def hindsight_store() -> HindsightStore:
    return HindsightStore()


@pytest.fixture
def always_pass_critic() -> MilestoneCritic:
    """Critic that always returns PASS for every milestone."""
    c = MilestoneCritic.__new__(MilestoneCritic)
    c._boundary = None
    c._pass_threshold = 0.0  # very permissive
    c._fail_threshold = -1.0  # impossible to fail
    c._require_violations_for_fail = False
    # Override critique to always pass
    def _always_pass(milestones, evidence_list):
        return [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.PASS,
                reasoning="Simulated PASS for testing",
                confidence=0.9,
            )
            for m in milestones
        ]
    c.critique = _always_pass
    c._critique_single = lambda m, e: MilestoneCriticResult(
        milestone_id=str(m.id),
        verdict=MilestoneVerdict.PASS,
        reasoning="Simulated PASS",
        confidence=0.9,
    )
    return c


@pytest.fixture
def always_fail_critic() -> MilestoneCritic:
    """Critic that always returns FAIL for every milestone."""
    c = MilestoneCritic.__new__(MilestoneCritic)
    def _always_fail(milestones, evidence_list):
        return [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Simulated FAIL for testing",
                confidence=0.9,
            )
            for m in milestones
        ]
    c.critique = _always_fail
    return c


@pytest.fixture
def partial_fail_critic() -> MilestoneCritic:
    """Critic that PASSes milestone 0, FAILs milestone 1."""
    def _partial_fail(milestones, evidence_list):
        results = []
        for i, m in enumerate(milestones):
            verdict = MilestoneVerdict.PASS if i == 0 else MilestoneVerdict.FAIL
            results.append(
                MilestoneCriticResult(
                    milestone_id=str(m.id),
                    verdict=verdict,
                    reasoning=f"Simulated {'PASS' if i == 0 else 'FAIL'} for milestone {i}",
                    confidence=0.9,
                )
            )
        return results
    c = MilestoneCritic.__new__(MilestoneCritic)
    c.critique = _partial_fail
    return c


def make_evidence(
    milestone_ids: list[str],
    confidence: float = 0.8,
) -> list[Evidence]:
    """Helper to create a list of evidence objects."""
    return [
        Evidence(
            milestone_id=mid,
            text_snippets=[f"Evidence for {mid}"],
            citations=[],
            intermediate_results=[],
            confidence=confidence,
        )
        for mid in milestone_ids
    ]


def simple_evidence_provider_factory(
    verdict_per_attempt: list[list[MilestoneVerdict]],
) -> callable:
    """
    Build an evidence provider that returns evidence keyed to a pre-set
    verdict sequence.

    verdict_per_attempt[attempt_idx][milestone_idx] = desired verdict
    The provider will cycle through the provided lists.
    """
    def provider(milestones, attempt, hindsight_entries):
        attempt_idx = min(attempt - 1, len(verdict_per_attempt) - 1)
        verdict_list = verdict_per_attempt[attempt_idx]
        return [
            Evidence(
                milestone_id=str(m.id),
                text_snippets=["Simulated evidence"],
                citations=[],
                intermediate_results=[],
                confidence=0.9 if v == MilestoneVerdict.PASS else 0.1,
            )
            for m, v in zip(milestones, verdict_list)
        ]
    return provider


# ---------------------------------------------------------------------------
# LoopResult dataclass tests
# ---------------------------------------------------------------------------


class TestLoopResult:
    def test_loop_result_fields(self):
        result = LoopResult(
            success=True,
            attempts=2,
            milestones_verified=[],
            milestone_verdicts=[],
            hindsight_entries=[],
            final_verdict="All passed",
        )
        assert result.success is True
        assert result.attempts == 2
        assert result.final_verdict == "All passed"

    def test_loop_result_defaults(self):
        result = LoopResult(success=False, attempts=1)
        assert result.milestones_verified == []
        assert result.milestone_verdicts == []
        assert result.hindsight_entries == []
        assert result.final_verdict == ""

    def test_loop_result_repr(self):
        result = LoopResult(success=True, attempts=1)
        assert "LoopResult" in repr(result)
        assert "success=True" in repr(result)


# ---------------------------------------------------------------------------
# _default_evidence_provider tests
# ---------------------------------------------------------------------------


class TestDefaultEvidenceProvider:
    def test_returns_empty_evidence_per_milestone(self):
        milestones = [
            Milestone("Task A", "spec A", "rubric A", order=0),
            Milestone("Task B", "spec B", "rubric B", order=1),
        ]
        evidence = _default_evidence_provider(milestones, 1, [])
        assert len(evidence) == 2
        assert all(e.confidence == 0.0 for e in evidence)
        assert all(e.text_snippets == [] for e in evidence)
        assert evidence[0].milestone_id == str(milestones[0].id)
        assert evidence[1].milestone_id == str(milestones[1].id)

    def test_empty_milestone_list(self):
        evidence = _default_evidence_provider([], 1, [])
        assert evidence == []


# ---------------------------------------------------------------------------
# HiMCALoop construction tests
# ---------------------------------------------------------------------------


class TestHiMCALoopConstruction:
    def test_default_construction(self):
        loop = HiMCALoop()
        assert loop.decomposer is not None
        assert loop.critic is not None
        assert loop.hindsight_store is not None
        assert isinstance(loop.hindsight_store, HindsightStore)

    def test_custom_components(self, decomposer, evidence_collector, critic, hindsight_store):
        loop = HiMCALoop(
            decomposer=decomposer,
            evidence_collector=evidence_collector,
            critic=critic,
            hindsight_store=hindsight_store,
        )
        assert loop.decomposer is decomposer
        assert loop.critic is critic
        # HindsightStore has no __eq__ so use repr identity check
        assert repr(loop.hindsight_store) == repr(hindsight_store)

    def test_custom_max_attempts(self):
        loop = HiMCALoop(max_attempts=5)
        result = loop.run("Test task", max_attempts=5)
        assert result.attempts <= 5

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            HiMCALoop(max_attempts=0)

    def test_run_max_attempts_must_be_positive(self):
        loop = HiMCALoop()
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            loop.run("Test task", max_attempts=0)


# ---------------------------------------------------------------------------
# HiMCALoop.run() — core path tests
# ---------------------------------------------------------------------------


class TestHiMCALoopRun:
    def test_run_empty_task_raises(self):
        loop = HiMCALoop()
        with pytest.raises(ValueError, match="Task description must not be empty"):
            loop.run("")

    def test_run_empty_whitespace_task_raises(self):
        loop = HiMCALoop()
        with pytest.raises(ValueError, match="Task description must not be empty"):
            loop.run("   ")

    def test_run_decomposes_task(self):
        loop = HiMCALoop()
        result = loop.run("Analyze revenue and generate a report", max_attempts=1)
        assert len(result.milestones_verified) >= 1

    def test_run_single_attempt_success(self, decomposer):
        """When all milestones pass on attempt 1, success=True, attempts=1."""
        always_pass = MilestoneCritic.__new__(MilestoneCritic)
        always_pass.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.PASS,
                reasoning="Always pass",
                confidence=0.9,
            )
            for m in ms
        ]

        loop = HiMCALoop(decomposer=decomposer, critic=always_pass, max_attempts=3)
        result = loop.run("Analyze revenue and generate a report", max_attempts=3)

        assert result.success is True
        assert result.attempts == 1
        assert result.final_verdict.startswith("All")

    def test_run_exhausts_attempts_on_failure(self, decomposer):
        """When no milestones pass, loop exhausts all max_attempts."""
        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]

        loop = HiMCALoop(decomposer=decomposer, critic=always_fail, max_attempts=3)
        result = loop.run("Analyze revenue and generate a report", max_attempts=3)

        assert result.success is False
        assert result.attempts == 3
        assert "FAIL" in result.final_verdict

    def test_run_early_termination_on_pass(self, decomposer):
        """Loop stops at attempt 2 if all milestones pass then."""
        # Mock critic: FAIL on attempt 1, PASS on attempt 2+
        verdict_cycle = [[MilestoneVerdict.FAIL], [MilestoneVerdict.PASS]]

        def mock_critique(milestones, evidence_list):
            attempt_idx = len([])  # will be tracked via closure
            # Use a simple counter via a list to track calls
            return [
                MilestoneCriticResult(
                    milestone_id=str(m.id),
                    verdict=MilestoneVerdict.PASS,
                    reasoning="Always pass",
                    confidence=0.9,
                )
                for m in milestones
            ]

        always_pass_critic = MilestoneCritic.__new__(MilestoneCritic)
        always_pass_critic.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.PASS,
                reasoning="Always pass",
                confidence=0.9,
            )
            for m in ms
        ]

        fail_then_pass_critic = MilestoneCritic.__new__(MilestoneCritic)
        call_count = [0]

        def fail_then_pass(ms, es):
            call_count[0] += 1
            verdicts = [MilestoneVerdict.FAIL] if call_count[0] == 1 else [MilestoneVerdict.PASS]
            return [
                MilestoneCriticResult(
                    milestone_id=str(m.id),
                    verdict=verdicts[0],
                    reasoning=f"Attempt {call_count[0]}: {verdicts[0].value}",
                    confidence=0.9,
                )
                for m in ms
            ]

        fail_then_pass_critic.critique = fail_then_pass

        loop = HiMCALoop(
            decomposer=decomposer,
            critic=fail_then_pass_critic,
            max_attempts=3,
        )
        result = loop.run("Analyze revenue", max_attempts=3)

        assert result.success is True
        assert result.attempts == 2

    def test_run_preserves_all_verdicts(self, decomposer):
        """All per-attempt verdicts are stored in milestone_verdicts."""
        verdict_list: list = []

        verdict_critic = MilestoneCritic.__new__(MilestoneCritic)
        verdict_list_index = [0]

        def multi_attempt_critique(ms, es):
            verdicts = [MilestoneVerdict.FAIL, MilestoneVerdict.PASS]
            v = verdicts[verdict_list_index[0] % len(verdicts)]
            verdict_list.append(v)
            verdict_list_index[0] += 1
            return [
                MilestoneCriticResult(
                    milestone_id=str(m.id),
                    verdict=v,
                    reasoning=f"Verdict: {v.value}",
                    confidence=0.9,
                )
                for m in ms
            ]

        verdict_critic.critique = multi_attempt_critique

        loop = HiMCALoop(
            decomposer=decomposer,
            critic=verdict_critic,
            max_attempts=3,
        )
        result = loop.run("Analyze revenue", max_attempts=3)

        # Should have stopped after 2 attempts (FAIL then PASS)
        assert len(result.milestone_verdicts) == 2
        assert result.milestone_verdicts[0][0].verdict == MilestoneVerdict.FAIL
        assert result.milestone_verdicts[1][0].verdict == MilestoneVerdict.PASS

    def test_run_stores_hindsight_entries(self, decomposer):
        """Failed milestones are stored in the hindsight store."""
        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]

        loop = HiMCALoop(decomposer=decomposer, critic=always_fail, max_attempts=2)
        result = loop.run("Analyze revenue", max_attempts=2)

        # Each failed milestone in each attempt → 2 entries (1 milestone × 2 attempts)
        assert len(result.hindsight_entries) >= 1
        assert all(
            e.critic_result.verdict in (MilestoneVerdict.FAIL, MilestoneVerdict.UNCERTAIN)
            for e in result.hindsight_entries
        )

    def test_run_hindsight_store_updated(self, decomposer):
        """The loop's hindsight store is updated after run()."""
        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]

        loop = HiMCALoop(decomposer=decomposer, critic=always_fail, max_attempts=2)
        loop.run("Analyze revenue", max_attempts=2)

        # The store should now contain entries
        assert len(loop.hindsight_store) >= 1

    def test_run_partial_failure_stores_only_failures(self, decomposer):
        """Only FAIL/UNCERTAIN milestones are stored as hindsight; PASS ones are skipped."""
        partial = MilestoneCritic.__new__(MilestoneCritic)
        partial.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.PASS if i == 0 else MilestoneVerdict.FAIL,
                reasoning="Partial fail",
                confidence=0.9,
            )
            for i, m in enumerate(ms)
        ]

        loop = HiMCALoop(decomposer=decomposer, critic=partial, max_attempts=1)
        result = loop.run("Analyze revenue and generate report", max_attempts=1)

        # Only the FAIL milestone should be in hindsight
        assert len(result.hindsight_entries) == 1

    def test_run_uncertain_is_stored(self, decomposer):
        """UNCERTAIN verdicts are also stored as hindsight entries."""
        uncertain_critic = MilestoneCritic.__new__(MilestoneCritic)
        uncertain_critic.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.UNCERTAIN,
                reasoning="Uncertain",
                confidence=0.5,
            )
            for m in ms
        ]

        loop = HiMCALoop(decomposer=decomposer, critic=uncertain_critic, max_attempts=1)
        result = loop.run("Analyze revenue", max_attempts=1)

        assert len(result.hindsight_entries) == len(result.milestones_verified)
        assert result.hindsight_entries[0].critic_result.verdict == MilestoneVerdict.UNCERTAIN

    def test_run_no_milestones_early_return(self, decomposer):
        """If task decomposes to no milestones, return early with failure."""
        # Use a decomposer that returns empty
        empty_decomposer = MilestoneDecomposer.__new__(MilestoneDecomposer)
        empty_decomposer.decompose = lambda t: []

        loop = HiMCALoop(decomposer=empty_decomposer, max_attempts=3)
        result = loop.run("Analyze revenue", max_attempts=3)

        assert result.success is False
        assert result.attempts == 0
        assert result.milestones_verified == []

    def test_run_evidence_provider_called_per_attempt(self, decomposer):
        """The evidence provider is called once per attempt."""
        call_count = {"count": 0}

        def counting_provider(milestones, attempt, hindsight_entries):
            call_count["count"] += 1
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

        loop = HiMCALoop(
            decomposer=decomposer,
            evidence_provider=counting_provider,
            max_attempts=3,
        )
        # Use always-fail critic so all attempts are used
        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]
        loop._critic = always_fail

        result = loop.run("Analyze revenue and generate a report", max_attempts=3)

        assert call_count["count"] == 3  # Called once per attempt

    def test_run_evidence_provider_receives_hindsight(self, decomposer):
        """The evidence provider receives accumulated hindsight entries."""
        received_hindsight: list = []

        def capturing_provider(milestones, attempt, hindsight_entries):
            received_hindsight.append(list(hindsight_entries))
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

        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]

        loop = HiMCALoop(
            decomposer=decomposer,
            evidence_provider=capturing_provider,
            critic=always_fail,
            max_attempts=2,
        )
        loop.run("Analyze revenue", max_attempts=2)

        # Attempt 2's provider should receive 1 hindsight entry from attempt 1
        assert len(received_hindsight[1]) == len(received_hindsight[0]) + 1

    def test_run_per_attempt_evidence_provider_override(self, decomposer):
        """Passing evidence_provider to run() overrides the instance-level one."""
        instance_call_count = {"count": 0}
        run_call_count = {"count": 0}

        def instance_provider(milestones, attempt, hindsight_entries):
            instance_call_count["count"] += 1
            return [
                Evidence(
                    milestone_id=str(m.id),
                    text_snippets=[],
                    citations=[],
                    intermediate_results=[],
                    confidence=0.9,
                )
                for m in milestones
            ]

        def run_provider(milestones, attempt, hindsight_entries):
            run_call_count["count"] += 1
            return [
                Evidence(
                    milestone_id=str(m.id),
                    text_snippets=[],
                    citations=[],
                    intermediate_results=[],
                    confidence=0.9,
                )
                for m in milestones
            ]

        loop = HiMCALoop(
            decomposer=decomposer,
            evidence_provider=instance_provider,
            max_attempts=1,
        )
        always_pass = MilestoneCritic.__new__(MilestoneCritic)
        always_pass.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.PASS,
                reasoning="Pass",
                confidence=0.9,
            )
            for m in ms
        ]
        loop._critic = always_pass

        loop.run("Analyze revenue", max_attempts=1, evidence_provider=run_provider)

        assert instance_call_count["count"] == 0  # Instance provider NOT called
        assert run_call_count["count"] == 1  # Run-level provider WAS called


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestHiMCALoopIntegration:
    def test_full_pipeline_with_real_components(self):
        """Run the loop with real decomposer, collector, and critic."""
        loop = HiMCALoop(max_attempts=2)

        # Real evidence provider that simulates a partial trajectory
        def trajectory_evidence_provider(milestones, attempt, hindsight_entries):
            return [
                Evidence(
                    milestone_id=str(m.id),
                    text_snippets=[
                        f"Evidence for: {m.description}"
                    ],
                    citations=[],
                    intermediate_results=[],
                    confidence=0.9 if attempt == 2 else 0.2,
                )
                for m in milestones
            ]

        result = loop.run(
            "Analyze revenue and generate a report",
            max_attempts=2,
            evidence_provider=trajectory_evidence_provider,
        )

        # Real critic with realistic thresholds should:
        # - Attempt 1: low confidence evidence → FAIL or UNCERTAIN
        # - Attempt 2: high confidence evidence → PASS (early exit)
        assert result.attempts == 2
        assert len(result.milestone_verdicts) == 2

    def test_multiple_milestones_all_pass_early(self):
        """Multiple milestones all passing on first attempt — verified by mock critic."""
        always_pass_critic = MilestoneCritic.__new__(MilestoneCritic)
        always_pass_critic.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.PASS,
                reasoning="Always pass",
                confidence=0.9,
            )
            for m in ms
        ]

        loop = HiMCALoop(max_attempts=3, critic=always_pass_critic)

        result = loop.run(
            "Analyze revenue, compare to last quarter, and generate a report",
            max_attempts=3,
        )

        assert result.success is True
        assert result.attempts == 1
        assert len(result.milestone_verdicts) == 1
        assert len(result.milestones_verified) >= 1

    def test_multiple_milestones_some_never_pass(self):
        """Multiple milestones, only some pass even after max_attempts."""
        loop = HiMCALoop(max_attempts=2)

        # Provider that always gives PASS to milestone 0 and FAIL to milestone 1
        def provider(milestones, attempt, hindsight):
            results = []
            for m in milestones:
                if m.order == 0:
                    results.append(
                        Evidence(
                            milestone_id=str(m.id),
                            text_snippets=["Pass evidence"],
                            citations=[],
                            intermediate_results=[],
                            confidence=0.9,
                        )
                    )
                else:
                    results.append(
                        Evidence(
                            milestone_id=str(m.id),
                            text_snippets=["Fail evidence"],
                            citations=[],
                            intermediate_results=[],
                            confidence=0.1,
                        )
                    )
            return results

        result = loop.run(
            "Analyze revenue and generate a report",
            max_attempts=2,
            evidence_provider=provider,
        )

        assert result.success is False
        assert result.attempts == 2
        assert "FAIL" in result.final_verdict

    def test_hindsight_entry_contains_original_task(self):
        """Hindsight entries capture the original task."""
        loop = HiMCALoop(max_attempts=1)

        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]
        loop._critic = always_fail

        task = "Analyze Q3 revenue"
        result = loop.run(task, max_attempts=1)

        assert len(result.hindsight_entries) >= 1
        assert result.hindsight_entries[0].original_task == task

    def test_hindsight_entry_episode_ids_unique(self):
        """Each run gets a unique episode_id."""
        loop = HiMCALoop(max_attempts=1)

        always_fail = MilestoneCritic.__new__(MilestoneCritic)
        always_fail.critique = lambda ms, es: [
            MilestoneCriticResult(
                milestone_id=str(m.id),
                verdict=MilestoneVerdict.FAIL,
                reasoning="Always fail",
                confidence=0.9,
            )
            for m in ms
        ]
        loop._critic = always_fail

        result1 = loop.run("Task 1", max_attempts=1)
        result2 = loop.run("Task 2", max_attempts=1)

        ids1 = {e.episode_id for e in result1.hindsight_entries}
        ids2 = {e.episode_id for e in result2.hindsight_entries}
        assert ids1.isdisjoint(ids2)
