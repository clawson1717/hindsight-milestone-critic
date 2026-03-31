"""
Tests for src.policy — HindsightAwarePolicy and PolicyDecision.

Step 6: Hindsight-Aware Policy
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.critic import BoundaryViolation, MilestoneCriticResult, MilestoneVerdict
from src.decompose import Milestone
from src.evidence import Evidence
from src.hindsight import HindsightEntry, HindsightStore
from src.policy import (
    HindsightAwarePolicy,
    PolicyDecision,
    _compute_keyword_overlap,
    _compute_relevance_score,
    _extract_keywords,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_store_dir() -> Path:
    """A temporary directory for the hindsight store."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def hindsight_store(temp_store_dir: Path) -> HindsightStore:
    """A fresh HindsightStore with a temporary directory."""
    return HindsightStore(store_dir=temp_store_dir)


@pytest.fixture
def sample_milestone() -> Milestone:
    """A milestone about analyzing revenue data."""
    return Milestone(
        description="Analyze Q3 revenue data",
        evidence_spec="Q3 revenue figures and growth rates",
        success_rubric="Pass if Q3 revenue growth rate is reported",
        order=0,
    )


@pytest.fixture
def sample_milestone_b() -> Milestone:
    """A milestone about generating a financial report."""
    return Milestone(
        description="Generate financial report",
        evidence_spec="Completed report document",
        success_rubric="Pass if report contains executive summary",
        order=1,
    )


@pytest.fixture
def sample_evidence(sample_milestone: Milestone) -> Evidence:
    """Evidence for sample_milestone."""
    return Evidence(
        milestone_id=str(sample_milestone.id),
        text_snippets=["Q3 revenue grew by 12% year-over-year"],
        citations=[],
        intermediate_results=[{"quarter": "Q3", "revenue": 4_200_000}],
        confidence=0.8,
    )


@pytest.fixture
def sample_critic_result_fail(sample_milestone: Milestone) -> MilestoneCriticResult:
    """A FAIL verdict for sample_milestone."""
    return MilestoneCriticResult(
        milestone_id=str(sample_milestone.id),
        verdict=MilestoneVerdict.FAIL,
        reasoning="Revenue growth rate was not computed correctly due to boundary violation: constraint overstep detected.",
        confidence=0.9,
        violations=[
            BoundaryViolation(
                violation_type="constraint_overstep",
                description="Incorrect growth rate calculation",
                severity="high",
                milestone_id=str(sample_milestone.id),
            ),
        ],
        evidence_quality=0.7,
        evidence_sufficiency=0.6,
        evidence_consistency=0.8,
    )


@pytest.fixture
def sample_trajectory() -> list[dict]:
    """A minimal 2-step trajectory."""
    return [
        {"step": 0, "role": "assistant", "content": "Analyzing Q3 revenue..."},
        {"step": 1, "role": "assistant", "content": "Computing year-over-year growth"},
    ]


def _make_simple_base_policy() -> callable:
    """Return a simple base policy for testing."""

    def policy(task_context: str) -> str:
        ctx_lower = task_context.lower()
        if "revenue" in ctx_lower:
            return "analyze_revenue"
        if "report" in ctx_lower:
            return "generate_report"
        if "cost" in ctx_lower:
            return "analyze_cost"
        return "default_action"

    return policy


@pytest.fixture
def simple_base_policy() -> callable:
    return _make_simple_base_policy()


# ---------------------------------------------------------------------------
# PolicyDecision tests
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    def test_construction_default(self) -> None:
        decision = PolicyDecision(action="test_action")
        assert decision.action == "test_action"
        assert decision.reasoning == ""
        assert decision.confidence == 1.0
        assert decision.hindsight_entries_used == []
        assert decision.shaped_context == ""

    def test_construction_full(self) -> None:
        entry = HindsightEntry(
            original_task="Test task",
            failed_milestone=Milestone(
                description="Test milestone",
                evidence_spec="Test evidence spec",
                success_rubric="Test rubric",
                order=0,
            ),
            failed_milestone_index=0,
            collected_evidence=Evidence(milestone_id="test"),
            critic_result=MilestoneCriticResult(
                milestone_id="test",
                verdict=MilestoneVerdict.FAIL,
                reasoning="Test reasoning",
                confidence=0.8,
            ),
            trajectory=[],
            hindsight_label="What if [test] was the actual goal?",
            retrieval_keywords=["test"],
        )
        decision = PolicyDecision(
            action="my_action",
            reasoning="Because it worked",
            confidence=0.75,
            hindsight_entries_used=[entry],
            shaped_context="Task context with hindsight",
        )
        assert decision.action == "my_action"
        assert decision.reasoning == "Because it worked"
        assert decision.confidence == 0.75
        assert len(decision.hindsight_entries_used) == 1
        assert decision.shaped_context == "Task context with hindsight"

    def test_to_dict(self) -> None:
        decision = PolicyDecision(
            action="test_action",
            reasoning="Test reasoning",
            confidence=0.9,
            hindsight_entries_used=[],
            shaped_context="context",
        )
        d = decision.to_dict()
        assert d["action"] == "test_action"
        assert d["reasoning"] == "Test reasoning"
        assert d["confidence"] == 0.9
        assert d["hindsight_entries_used"] == []
        assert d["shaped_context"] == "context"

    def test_from_dict(self) -> None:
        raw = {
            "action": "from_dict_action",
            "reasoning": "From dict reasoning",
            "confidence": 0.6,
            "hindsight_entries_used": [],
            "shaped_context": "shaped",
        }
        decision = PolicyDecision.from_dict(raw)
        assert decision.action == "from_dict_action"
        assert decision.confidence == 0.6

    def test_repr(self) -> None:
        decision = PolicyDecision(action="my_test_action_1234567890")
        r = repr(decision)
        assert "PolicyDecision" in r
        assert "my_test_action" in r


# ---------------------------------------------------------------------------
# Relevance scoring utility tests
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_extracts_alpha_tokens(self) -> None:
        result = _extract_keywords("Analyze Q3 revenue data carefully")
        assert "analyze" in result
        assert "revenue" in result
        assert "data" in result
        assert "carefully" in result

    def test_strips_stopwords(self) -> None:
        result = _extract_keywords("the and for but are not you all")
        assert len(result) == 0

    def test_case_insensitive(self) -> None:
        result = _extract_keywords("REVENUE ANALYZE")
        assert "revenue" in result
        assert "analyze" in result

    def test_empty_string(self) -> None:
        assert _extract_keywords("") == []
        assert _extract_keywords("the and a") == []

    def test_min_len(self) -> None:
        result = _extract_keywords("an it on", min_len=3)
        assert result == []


class TestKeywordOverlap:
    def test_identical_lists(self) -> None:
        keywords = ["revenue", "analysis", "data"]
        score = _compute_keyword_overlap(keywords, keywords)
        assert abs(score - 1.0) < 1e-9

    def test_disjoint_lists(self) -> None:
        score = _compute_keyword_overlap(["a", "b"], ["c", "d"])
        assert abs(score - 0.0) < 1e-9

    def test_partial_overlap(self) -> None:
        score = _compute_keyword_overlap(["a", "b", "c"], ["b", "c", "d"])
        # intersection = {b, c} = 2, union = {a, b, c, d} = 4
        assert abs(score - 0.5) < 1e-9

    def test_empty_first(self) -> None:
        assert _compute_keyword_overlap([], ["a", "b"]) == 0.0

    def test_empty_second(self) -> None:
        assert _compute_keyword_overlap(["a", "b"], []) == 0.0

    def test_both_empty(self) -> None:
        assert _compute_keyword_overlap([], []) == 0.0


class TestComputeRelevanceScore:
    def test_perfect_task_match(self) -> None:
        entry = HindsightEntry(
            original_task="Analyze Q3 revenue data",
            failed_milestone=Milestone(
                description="Analyze revenue",
                evidence_spec="Revenue figures",
                success_rubric="Pass if analyzed",
                order=0,
            ),
            failed_milestone_index=0,
            collected_evidence=Evidence(milestone_id="test"),
            critic_result=MilestoneCriticResult(
                milestone_id="test",
                verdict=MilestoneVerdict.FAIL,
                reasoning="Failed",
                confidence=0.8,
            ),
            trajectory=[],
            hindsight_label="What if [...]?",
            retrieval_keywords=["revenue", "analyze"],
        )
        score = _compute_relevance_score("Analyze Q3 revenue data", entry)
        assert 0.0 <= score <= 1.0
        # Perfect match → high score
        assert score > 0.5

    def test_no_match(self) -> None:
        entry = HindsightEntry(
            original_task="Weather forecast task",
            failed_milestone=Milestone(
                description="Get weather data",
                evidence_spec="Weather data",
                success_rubric="Pass",
                order=0,
            ),
            failed_milestone_index=0,
            collected_evidence=Evidence(milestone_id="test"),
            critic_result=MilestoneCriticResult(
                milestone_id="test",
                verdict=MilestoneVerdict.FAIL,
                reasoning="Failed",
                confidence=0.8,
            ),
            trajectory=[],
            hindsight_label="What if [...]?",
            retrieval_keywords=["weather"],
        )
        score = _compute_relevance_score("Analyze Q3 revenue data", entry)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# HindsightAwarePolicy initialization tests
# ---------------------------------------------------------------------------


class TestHindsightAwarePolicyInit:
    def test_default_init(self, hindsight_store: HindsightStore, simple_base_policy: callable) -> None:
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        assert policy._top_k == 5
        assert policy._relevance_threshold == 0.0
        assert policy._shaping_method == "append"
        assert policy._store is hindsight_store
        assert policy._base_policy is simple_base_policy

    def test_custom_init(self, hindsight_store: HindsightStore, simple_base_policy: callable) -> None:
        policy = HindsightAwarePolicy(
            simple_base_policy,
            hindsight_store,
            top_k=3,
            relevance_threshold=0.3,
            shaping_method="prepend",
        )
        assert policy._top_k == 3
        assert policy._relevance_threshold == 0.3
        assert policy._shaping_method == "prepend"

    def test_negative_top_k_raises(self, hindsight_store: HindsightStore, simple_base_policy: callable) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 0"):
            HindsightAwarePolicy(simple_base_policy, hindsight_store, top_k=-1)

    def test_invalid_relevance_threshold_raises(self, hindsight_store: HindsightStore, simple_base_policy: callable) -> None:
        with pytest.raises(ValueError, match="relevance_threshold must be in"):
            HindsightAwarePolicy(simple_base_policy, hindsight_store, relevance_threshold=1.5)

    def test_invalid_shaping_method_raises(self, hindsight_store: HindsightStore, simple_base_policy: callable) -> None:
        with pytest.raises(ValueError, match="shaping_method must be"):
            HindsightAwarePolicy(simple_base_policy, hindsight_store, shaping_method="invalid")

    def test_repr(self, hindsight_store: HindsightStore, simple_base_policy: callable) -> None:
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store, top_k=3)
        r = repr(policy)
        assert "HindsightAwarePolicy" in r
        assert "top_k=3" in r


# ---------------------------------------------------------------------------
# HindsightAwarePolicy.get_action tests
# ---------------------------------------------------------------------------


class TestHindsightAwarePolicyGetAction:
    def test_get_action_empty_store(
        self, hindsight_store: HindsightStore, simple_base_policy: callable
    ) -> None:
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        decision = policy.get_action("Analyze Q3 revenue data")
        assert decision.action == "analyze_revenue"
        assert len(decision.hindsight_entries_used) == 0
        assert decision.confidence == 0.5  # No hindsight entries → neutral
        assert decision.shaped_context == "Analyze Q3 revenue data"

    def test_get_action_with_hindsight_entries(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        # Populate the store
        hindsight_store.add(
            task="Analyze Q3 revenue",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store, top_k=3)
        decision = policy.get_action("Analyze Q3 revenue and generate report")

        assert decision.action == "analyze_revenue"
        assert len(decision.hindsight_entries_used) == 1
        assert decision.hindsight_entries_used[0].original_task == "Analyze Q3 revenue"
        assert "HINDSIGHT CONTEXT" in decision.shaped_context
        assert decision.confidence > 0.0

    def test_get_action_with_milestones(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        milestones = [sample_milestone]
        decision = policy.get_action("Analyze Q3 revenue data", milestones=milestones)
        assert decision.action == "analyze_revenue"
        assert len(decision.hindsight_entries_used) >= 0

    def test_get_action_shaping_method_prepend(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(
            simple_base_policy, hindsight_store, shaping_method="prepend"
        )
        decision = policy.get_action("Analyze revenue data")
        # With prepend, hindsight section should appear before the original task
        assert "## HINDSIGHT CONTEXT" in decision.shaped_context
        # The original task should appear after the hindsight section
        task_pos = decision.shaped_context.find("Analyze revenue data")
        hindsight_pos = decision.shaped_context.find("## HINDSIGHT CONTEXT")
        assert hindsight_pos < task_pos, "Hindsight section should appear before original task"

    def test_get_action_shaping_method_append(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(
            simple_base_policy, hindsight_store, shaping_method="append"
        )
        decision = policy.get_action("Analyze revenue data")
        # With append, original task should be at the start
        assert decision.shaped_context.startswith("Analyze revenue data")
        assert "## HINDSIGHT CONTEXT" in decision.shaped_context

    def test_get_action_top_k_limit(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        # Add multiple entries
        hindsight_store.add(
            task="Revenue analysis task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        hindsight_store.add(
            task="Cost analysis task",
            milestones=[sample_milestone_b],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store, top_k=1)
        decision = policy.get_action("revenue analysis")
        assert len(decision.hindsight_entries_used) <= 1

    def test_get_action_relevance_threshold_filters(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Weather forecast",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(
            simple_base_policy, hindsight_store, relevance_threshold=0.99
        )
        decision = policy.get_action("Analyze revenue")
        # Unrelated task with high threshold → should get no entries
        assert len(decision.hindsight_entries_used) == 0

    def test_get_action_base_policy_error_fallback(
        self, hindsight_store: HindsightStore, simple_base_policy: callable
    ) -> None:
        def error_policy(task_context: str) -> str:
            raise RuntimeError("Policy error")

        policy = HindsightAwarePolicy(error_policy, hindsight_store)
        decision = policy.get_action("Any task")
        assert "fallback_action_due_to_error" in decision.action
        assert decision.confidence == 0.0

    def test_get_action_reasoning_contains_entries(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        decision = policy.get_action("Analyze revenue")
        assert len(decision.reasoning) > 0
        assert "failure" in decision.reasoning.lower()

    def test_get_action_shaped_context_contains_hindsight_label(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        decision = policy.get_action("Analyze revenue")
        assert "What if" in decision.shaped_context
        assert "HINDSIGHT CONTEXT" in decision.shaped_context


# ---------------------------------------------------------------------------
# HindsightAwarePolicy.shape_prompt tests
# ---------------------------------------------------------------------------


class TestHindsightAwarePolicyShapePrompt:
    def test_shape_prompt_empty_entries(
        self, hindsight_store: HindsightStore, simple_base_policy: callable
    ) -> None:
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        result = policy.shape_prompt("Analyze revenue", [])
        assert result == "Analyze revenue"

    def test_shape_prompt_single_entry(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        entries = hindsight_store.retrieve("revenue analysis")
        result = policy.shape_prompt("Analyze revenue", entries)

        assert result.startswith("Analyze revenue")
        assert "HINDSIGHT CONTEXT" in result
        assert "END HINDSIGHT CONTEXT" in result
        assert "Previous Failure 1" in result
        assert "Revenue analysis" in result
        assert "How to avoid" in result

    def test_shape_prompt_multiple_entries(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        hindsight_store.add(
            task="Cost analysis",
            milestones=[sample_milestone_b],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        entries = hindsight_store.retrieve("analysis", top_k=5)
        result = policy.shape_prompt("General analysis task", entries)

        assert "Previous Failure 1" in result
        assert "Previous Failure 2" in result


# ---------------------------------------------------------------------------
# HindsightAwarePolicy.get_relevant_hindsight tests
# ---------------------------------------------------------------------------


class TestHindsightAwarePolicyGetRelevantHindsight:
    def test_get_relevant_hindsight_empty_store(
        self, hindsight_store: HindsightStore, simple_base_policy: callable
    ) -> None:
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        result = policy.get_relevant_hindsight("Any task")
        assert result == []

    def test_get_relevant_hindsight_with_entries(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        result = policy.get_relevant_hindsight("revenue analysis")
        assert len(result) == 1
        assert result[0].original_task == "Revenue analysis task"

    def test_get_relevant_hindsight_top_k_override(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        hindsight_store.add(
            task="Cost analysis",
            milestones=[sample_milestone_b],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store, top_k=5)
        result = policy.get_relevant_hindsight("analysis", top_k=1)
        assert len(result) <= 1


# ---------------------------------------------------------------------------
# HindsightAwarePolicy.relevance_scores tests
# ---------------------------------------------------------------------------


class TestHindsightAwarePolicyRelevanceScores:
    def test_relevance_scores_empty(
        self, hindsight_store: HindsightStore, simple_base_policy: callable
    ) -> None:
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        scores = policy.relevance_scores("Analyze revenue", [])
        assert scores == []

    def test_relevance_scores_with_entries(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        entries = hindsight_store.retrieve("revenue analysis")
        scores = policy.relevance_scores("Analyze Q3 revenue", entries)
        assert len(scores) == len(entries)
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# Integration with MilestoneCritic
# ---------------------------------------------------------------------------


class TestHindsightAwarePolicyIntegration:
    def test_end_to_end_with_milestone_critic(
        self,
        hindsight_store: HindsightStore,
        simple_base_policy: callable,
    ) -> None:
        """
        Integration test: decompose a task, run critic, add to store,
        then use HindsightAwarePolicy to get a shaped decision.
        """
        from src.critic import MilestoneCritic
        from src.decompose import MilestoneDecomposer
        from src.evidence import EvidenceCollector

        decomposer = MilestoneDecomposer()
        collector = EvidenceCollector()
        critic = MilestoneCritic()

        # First episode: task that will fail
        task_fail = "Analyze Q3 revenue data and compute the growth rate"
        milestones_fail = decomposer.decompose(task_fail)

        trajectory_fail = [
            {"action": "analyze Q3 revenue", "observation": "Revenue found", "thought": "...", "timestamp": "2024-01-01T00:00:00Z"},
            {"action": "compute growth rate", "observation": "Growth rate computed incorrectly", "thought": "...", "timestamp": "2024-01-01T00:01:00Z"},
        ]

        evidence_list_fail = [collector.collect(m, trajectory_fail) for m in milestones_fail]
        results_fail = critic.critique(milestones_fail, evidence_list_fail)

        # Add failed milestone to store
        hindsight_store.add(
            task=task_fail,
            milestones=milestones_fail,
            evidence_list=evidence_list_fail,
            critic_results=results_fail,
            trajectory=trajectory_fail,
        )

        # Now run the same task again using HindsightAwarePolicy
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store, top_k=3)
        decision = policy.get_action(task_fail, milestones_fail)

        assert isinstance(decision.action, str)
        assert isinstance(decision.confidence, float)
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.hindsight_entries_used) >= 1

        # The shaped context should include the hindsight information
        assert "HINDSIGHT CONTEXT" in decision.shaped_context
        assert "What if" in decision.shaped_context

        # Verify the hindsight entry has correct verdict
        for entry in decision.hindsight_entries_used:
            assert entry.critic_result.verdict in (
                MilestoneVerdict.FAIL,
                MilestoneVerdict.UNCERTAIN,
            )

    def test_hindsight_preserves_violations(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Test task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        decision = policy.get_action("Test task")

        for entry in decision.hindsight_entries_used:
            assert len(entry.critic_result.violations) >= 1
            high_severity = [v for v in entry.critic_result.violations if v.severity == "high"]
            assert len(high_severity) >= 1


# ---------------------------------------------------------------------------
# PolicyDecision roundtrip
# ---------------------------------------------------------------------------


class TestPolicyDecisionRoundtrip:
    def test_to_dict_from_dict_roundtrip(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        simple_base_policy: callable,
    ) -> None:
        hindsight_store.add(
            task="Test task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        decision = policy.get_action("Test task")

        d = decision.to_dict()
        restored = PolicyDecision.from_dict(d)

        assert restored.action == decision.action
        assert restored.reasoning == decision.reasoning
        assert restored.confidence == decision.confidence
        assert len(restored.hindsight_entries_used) == len(decision.hindsight_entries_used)
        assert restored.shaped_context == decision.shaped_context


# ---------------------------------------------------------------------------
# Avoidance guidance derivation
# ---------------------------------------------------------------------------


class TestAvoidanceGuidance:
    def test_derives_insufficient_guidance(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        simple_base_policy: callable,
    ) -> None:
        # Create a critic result with "insufficient" in reasoning
        result = MilestoneCriticResult(
            milestone_id=str(sample_milestone.id),
            verdict=MilestoneVerdict.FAIL,
            reasoning="Evidence is insufficient and incomplete",
            confidence=0.7,
            violations=[],
        )
        entry = HindsightEntry(
            original_task="Test task",
            failed_milestone=sample_milestone,
            failed_milestone_index=0,
            collected_evidence=Evidence(milestone_id=str(sample_milestone.id)),
            critic_result=result,
            trajectory=[],
            hindsight_label="What if [...]?",
            retrieval_keywords=["test"],
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        guidance = policy._derive_avoidance_guidance(entry)
        assert "evidence" in guidance.lower() or "sufficient" in guidance.lower()

    def test_derives_constraint_violation_guidance(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        simple_base_policy: callable,
    ) -> None:
        result = MilestoneCriticResult(
            milestone_id=str(sample_milestone.id),
            verdict=MilestoneVerdict.FAIL,
            reasoning="Boundary violation: constraint overstep detected",
            confidence=0.7,
            violations=[],
        )
        entry = HindsightEntry(
            original_task="Test task",
            failed_milestone=sample_milestone,
            failed_milestone_index=0,
            collected_evidence=Evidence(milestone_id=str(sample_milestone.id)),
            critic_result=result,
            trajectory=[],
            hindsight_label="What if [...]?",
            retrieval_keywords=["test"],
        )
        policy = HindsightAwarePolicy(simple_base_policy, hindsight_store)
        guidance = policy._derive_avoidance_guidance(entry)
        assert "constraint" in guidance.lower() or "boundary" in guidance.lower()
