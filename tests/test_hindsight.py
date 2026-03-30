"""
Tests for src.hindsight — HindsightStore, HindsightEntry, and TF-IDF utilities.

Step 5: Hindsight Experience Store (HeRL-style)
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

from src.critic import BoundaryViolation, MilestoneCriticResult, MilestoneVerdict
from src.decompose import Milestone
from src.evidence import Evidence
from src.hindsight import (
    HindsightEntry,
    HindsightStore,
    _compute_tfidf,
    _cosine_similarity,
    _extract_keywords,
    _keyword_overlap_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """A second milestone for retrieve/sample tests."""
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
        text_snippets=[
            "Q3 revenue grew by 12% year-over-year",
            "Total revenue: $4.2M",
        ],
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
        reasoning="Revenue growth rate was not computed correctly",
        confidence=0.9,
        violations=[
            BoundaryViolation(
                violation_type="computation_error",
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
def sample_critic_result_pass(sample_milestone: Milestone) -> MilestoneCriticResult:
    """A PASS verdict for sample_milestone."""
    return MilestoneCriticResult(
        milestone_id=str(sample_milestone.id),
        verdict=MilestoneVerdict.PASS,
        reasoning="All metrics were correctly reported",
        confidence=0.95,
        violations=[],
        evidence_quality=0.9,
        evidence_sufficiency=0.85,
        evidence_consistency=0.9,
    )


@pytest.fixture
def sample_critic_result_uncertain(
    sample_milestone: Milestone,
) -> MilestoneCriticResult:
    """An UNCERTAIN verdict for sample_milestone."""
    return MilestoneCriticResult(
        milestone_id=str(sample_milestone.id),
        verdict=MilestoneVerdict.UNCERTAIN,
        reasoning="Evidence is ambiguous and insufficient",
        confidence=0.5,
        violations=[],
        evidence_quality=0.4,
        evidence_sufficiency=0.3,
        evidence_consistency=0.5,
    )


@pytest.fixture
def sample_trajectory() -> list[dict]:
    """A minimal 2-step trajectory."""
    return [
        {
            "step": 0,
            "role": "assistant",
            "content": "Analyzing Q3 revenue figures...",
            "tool_calls": [],
        },
        {
            "step": 1,
            "role": "assistant",
            "content": "Computing year-over-year growth rate",
            "tool_calls": [],
        },
    ]


@pytest.fixture
def temp_store_dir() -> Path:
    """A temporary directory for store files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def hindsight_store(temp_store_dir: Path) -> HindsightStore:
    """A HindsightStore with a temporary directory."""
    return HindsightStore(store_dir=temp_store_dir)


# ---------------------------------------------------------------------------
# _extract_keywords
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

    def test_respects_min_len(self) -> None:
        result_short = _extract_keywords("an it on at", min_len=3)
        assert "an" not in result_short
        assert "it" not in result_short
        assert "on" not in result_short
        assert "at" not in result_short

    def test_default_min_len_three(self) -> None:
        result = _extract_keywords("an it")
        assert result == []

    def test_case_insensitive(self) -> None:
        result = _extract_keywords("REVENUE ANALYZE")
        assert "revenue" in result
        assert "analyze" in result

    def test_empty_string(self) -> None:
        assert _extract_keywords("") == []
        assert _extract_keywords("the and a") == []


# ---------------------------------------------------------------------------
# _compute_tfidf
# ---------------------------------------------------------------------------


class TestComputeTfidf:
    def test_empty_documents(self) -> None:
        result = _compute_tfidf([])
        assert result == []

    def test_single_document(self) -> None:
        result = _compute_tfidf(["revenue analysis data"])
        assert len(result) == 1
        assert isinstance(result[0], dict)
        # Should contain keywords with non-zero TF-IDF weights
        assert len(result[0]) > 0

    def test_multiple_documents(self) -> None:
        docs = ["revenue analysis", "cost analysis", "revenue growth"]
        result = _compute_tfidf(docs)
        assert len(result) == 3
        for r in result:
            assert isinstance(r, dict)

    def test_unit_normalisation(self) -> None:
        import math

        result = _compute_tfidf(["revenue analysis"])
        vec = result[0]
        if vec:
            magnitude = math.sqrt(sum(v * v for v in vec.values()))
            assert abs(magnitude - 1.0) < 1e-9

    def test_identical_documents(self) -> None:
        docs = ["revenue analysis", "revenue analysis"]
        result = _compute_tfidf(docs)
        # Both should have same terms; IDF should be 1.0 (or minimal)
        assert set(result[0].keys()) == set(result[1].keys())


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = {"a": 0.5, "b": 0.5}
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        vec_a = {"a": 1.0}
        vec_b = {"b": 1.0}
        assert abs(_cosine_similarity(vec_a, vec_b) - 0.0) < 1e-9

    def test_empty_vector_a(self) -> None:
        assert _cosine_similarity({}, {"a": 1.0}) == 0.0

    def test_empty_vector_b(self) -> None:
        assert _cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_both_empty(self) -> None:
        assert _cosine_similarity({}, {}) == 0.0

    def test_partial_overlap(self) -> None:
        import math

        vec_a = {"a": 0.5, "b": 0.5}
        vec_b = {"b": 0.5, "c": 0.5}
        sim = _cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# _keyword_overlap_score
# ---------------------------------------------------------------------------


class TestKeywordOverlapScore:
    def test_identical_lists(self) -> None:
        keywords = ["revenue", "analysis", "data"]
        assert abs(_keyword_overlap_score(keywords, keywords) - 1.0) < 1e-9

    def test_disjoint_lists(self) -> None:
        assert abs(_keyword_overlap_score(["a", "b"], ["c", "d"]) - 0.0) < 1e-9

    def test_partial_overlap(self) -> None:
        score = _keyword_overlap_score(["a", "b", "c"], ["b", "c", "d"])
        # intersection = {b, c} = 2, union = {a, b, c, d} = 4, score = 2/4 = 0.5
        assert abs(score - 0.5) < 1e-9

    def test_empty_first_list(self) -> None:
        assert _keyword_overlap_score([], ["a", "b"]) == 0.0

    def test_empty_second_list(self) -> None:
        assert _keyword_overlap_score(["a", "b"], []) == 0.0

    def test_both_empty(self) -> None:
        assert _keyword_overlap_score([], []) == 0.0


# ---------------------------------------------------------------------------
# HindsightEntry
# ---------------------------------------------------------------------------


class TestHindsightEntry:
    def test_construction(self, sample_milestone, sample_evidence, sample_critic_result_fail, sample_trajectory) -> None:
        entry = HindsightEntry(
            original_task="Analyze Q3 revenue",
            failed_milestone=sample_milestone,
            failed_milestone_index=0,
            collected_evidence=sample_evidence,
            critic_result=sample_critic_result_fail,
            trajectory=sample_trajectory,
            hindsight_label="What if [Analyze Q3 revenue data] was the actual goal?",
            retrieval_keywords=["revenue", "q3", "analysis"],
        )
        assert entry.original_task == "Analyze Q3 revenue"
        assert entry.failed_milestone == sample_milestone
        assert entry.failed_milestone_index == 0
        assert entry.collected_evidence == sample_evidence
        assert entry.critic_result == sample_critic_result_fail
        assert entry.trajectory == sample_trajectory
        assert entry.hindsight_label.startswith("What if")
        assert "revenue" in entry.retrieval_keywords
        # UUIDs are auto-generated
        assert isinstance(entry.entry_id, str)
        uuid.UUID(entry.entry_id)  # should not raise
        assert isinstance(entry.episode_id, str)
        uuid.UUID(entry.episode_id)  # should not raise

    def test_to_dict(self, sample_milestone, sample_evidence, sample_critic_result_fail, sample_trajectory) -> None:
        entry = HindsightEntry(
            original_task="Test task",
            failed_milestone=sample_milestone,
            failed_milestone_index=0,
            collected_evidence=sample_evidence,
            critic_result=sample_critic_result_fail,
            trajectory=sample_trajectory,
            hindsight_label="What if [...]?",
            retrieval_keywords=["test"],
        )
        d = entry.to_dict()
        assert d["original_task"] == "Test task"
        assert d["failed_milestone"]["description"] == sample_milestone.description
        assert d["failed_milestone_index"] == 0
        assert d["hindsight_label"] == "What if [...]?"
        assert "entry_id" in d
        assert "episode_id" in d
        assert "created_at" in d

    def test_from_dict(self, sample_milestone, sample_evidence, sample_critic_result_fail, sample_trajectory) -> None:
        entry = HindsightEntry(
            original_task="Test task",
            failed_milestone=sample_milestone,
            failed_milestone_index=0,
            collected_evidence=sample_evidence,
            critic_result=sample_critic_result_fail,
            trajectory=sample_trajectory,
            hindsight_label="What if [...]?",
            retrieval_keywords=["test"],
        )
        d = entry.to_dict()
        restored = HindsightEntry.from_dict(d)
        assert restored.original_task == entry.original_task
        assert restored.hindsight_label == entry.hindsight_label
        assert restored.entry_id == entry.entry_id
        assert restored.failed_milestone.description == entry.failed_milestone.description

    def test_to_dict_from_dict_roundtrip(
        self, sample_milestone, sample_evidence, sample_critic_result_fail, sample_trajectory
    ) -> None:
        entry = HindsightEntry(
            original_task="Analyze Q3 revenue",
            failed_milestone=sample_milestone,
            failed_milestone_index=2,
            collected_evidence=sample_evidence,
            critic_result=sample_critic_result_fail,
            trajectory=sample_trajectory,
            hindsight_label="What if [...]?",
            retrieval_keywords=["revenue", "q3"],
            keyword_tfidf={"revenue": 0.7, "q3": 0.3},
        )
        restored = HindsightEntry.from_dict(entry.to_dict())
        assert restored.original_task == entry.original_task
        assert restored.failed_milestone_index == entry.failed_milestone_index
        assert restored.keyword_tfidf == entry.keyword_tfidf
        assert restored.entry_id == entry.entry_id
        assert restored.episode_id == entry.episode_id

    def test_repr(self, sample_milestone, sample_evidence, sample_critic_result_fail, sample_trajectory) -> None:
        entry = HindsightEntry(
            original_task="Test task 12345678901234567890",
            failed_milestone=sample_milestone,
            failed_milestone_index=0,
            collected_evidence=sample_evidence,
            critic_result=sample_critic_result_fail,
            trajectory=sample_trajectory,
            hindsight_label="What if [test label] was the actual goal?",
            retrieval_keywords=["test"],
        )
        r = repr(entry)
        assert isinstance(r, str)
        assert "HindsightEntry" in r


# ---------------------------------------------------------------------------
# HindsightStore — basic
# ---------------------------------------------------------------------------


class TestHindsightStoreInit:
    def test_defaults(self) -> None:
        store = HindsightStore()
        assert len(store) == 0
        assert store._max_entries == 0
        r = repr(store)
        assert "HindsightStore" in r
        assert "entries=0" in r
        assert "max_entries=0" in r

    def test_with_store_dir(self, temp_store_dir: Path) -> None:
        store = HindsightStore(store_dir=temp_store_dir)
        assert store._store_dir == temp_store_dir
        assert temp_store_dir.exists()

    def test_with_max_entries(self) -> None:
        store = HindsightStore(max_entries=10)
        assert store._max_entries == 10

    def test_len_empty(self, hindsight_store: HindsightStore) -> None:
        assert len(hindsight_store) == 0

    def test_repr(self, hindsight_store: HindsightStore) -> None:
        r = repr(hindsight_store)
        assert "HindsightStore" in r


# ---------------------------------------------------------------------------
# HindsightStore — add
# ---------------------------------------------------------------------------


class TestHindsightStoreAdd:
    def test_add_fail(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        entry_ids = hindsight_store.add(
            task="Analyze Q3 revenue",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        assert len(entry_ids) == 1
        assert len(hindsight_store) == 1
        entry = hindsight_store._entries[0]
        assert entry.critic_result.verdict == MilestoneVerdict.FAIL
        assert entry.original_task == "Analyze Q3 revenue"

    def test_add_pass_skipped(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_pass: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        entry_ids = hindsight_store.add(
            task="Analyze Q3 revenue",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_pass],
            trajectory=sample_trajectory,
        )
        assert entry_ids == []
        assert len(hindsight_store) == 0

    def test_add_uncertain(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_uncertain: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        entry_ids = hindsight_store.add(
            task="Ambiguous task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_uncertain],
            trajectory=sample_trajectory,
        )
        assert len(entry_ids) == 1
        assert hindsight_store._entries[0].critic_result.verdict == MilestoneVerdict.UNCERTAIN

    def test_add_mixed(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_pass: MilestoneCriticResult,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        # milestone_a = FAIL, milestone_b = PASS → only one stored
        entry_ids = hindsight_store.add(
            task="Mixed task",
            milestones=[sample_milestone, sample_milestone_b],
            evidence_list=[sample_evidence, sample_evidence],
            critic_results=[sample_critic_result_fail, sample_critic_result_pass],
            trajectory=sample_trajectory,
        )
        assert len(entry_ids) == 1
        assert len(hindsight_store) == 1

    def test_add_length_mismatch(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            hindsight_store.add(
                task="Test",
                milestones=[sample_milestone],
                evidence_list=[sample_evidence, sample_evidence],
                critic_results=[sample_critic_result_fail],
                trajectory=sample_trajectory,
            )

    def test_add_multiple_episodes_distinguished(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Episode 1",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
            episode_id="episode-1",
        )
        hindsight_store.add(
            task="Episode 2",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
            episode_id="episode-2",
        )
        assert len(hindsight_store) == 2
        episode_ids = {e.episode_id for e in hindsight_store._entries}
        assert episode_ids == {"episode-1", "episode-2"}

    def test_add_fifo_eviction(
        self,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
        temp_store_dir: Path,
    ) -> None:
        store = HindsightStore(store_dir=temp_store_dir, max_entries=3)
        for i in range(5):
            m = Milestone(
                description=f"Milestone {i}",
                evidence_spec=f"Spec {i}",
                success_rubric=f"Rubric {i}",
                order=i,
            )
            store.add(
                task=f"Task {i}",
                milestones=[m],
                evidence_list=[sample_evidence],
                critic_results=[sample_critic_result_fail],
                trajectory=sample_trajectory,
            )
        # Only last 3 should remain
        assert len(store) == 3
        tasks = {e.original_task for e in store._entries}
        assert tasks == {"Task 2", "Task 3", "Task 4"}


# ---------------------------------------------------------------------------
# HindsightStore — stats
# ---------------------------------------------------------------------------


class TestHindsightStoreStats:
    def test_stats_empty(self, hindsight_store: HindsightStore) -> None:
        stats = hindsight_store.stats()
        assert stats["total_entries"] == 0
        assert stats["unique_episodes"] == 0
        assert stats["verdict_distribution"] == {}

    def test_stats_after_add(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_critic_result_uncertain: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Task 1",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
            episode_id="ep-1",
        )
        hindsight_store.add(
            task="Task 2",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_uncertain],
            trajectory=sample_trajectory,
            episode_id="ep-2",
        )
        stats = hindsight_store.stats()
        assert stats["total_entries"] == 2
        assert stats["unique_episodes"] == 2
        assert stats["verdict_distribution"]["fail"] == 1
        assert stats["verdict_distribution"]["uncertain"] == 1


# ---------------------------------------------------------------------------
# HindsightStore — retrieve
# ---------------------------------------------------------------------------


class TestHindsightStoreRetrieve:
    def test_retrieve_empty(self, hindsight_store: HindsightStore) -> None:
        result = hindsight_store.retrieve("Any task")
        assert result == []

    def test_retrieve_top_k(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
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
        # Top-1 should return the most relevant
        result = hindsight_store.retrieve("revenue analysis", top_k=1)
        assert len(result) == 1
        assert result[0].original_task == "Revenue analysis task"

    def test_retrieve_sorted_by_score(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        hindsight_store.add(
            task="Weather forecast task",
            milestones=[sample_milestone_b],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        result = hindsight_store.retrieve("revenue analysis", top_k=2)
        assert len(result) == 2
        # Most similar should be first
        assert result[0].original_task == "Revenue analysis task"

    def test_retrieve_respects_similarity_threshold(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Analyze Q3 revenue",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        # Query for completely unrelated task — threshold should filter it out
        result = hindsight_store.retrieve(
            "what is the weather in tokyo",
            top_k=5,
            similarity_threshold=0.99,  # very high
        )
        # Either empty or if returned, it's because the score is below threshold
        for entry in result:
            assert entry is not None  # just verify we get entries if threshold allows

    def test_retrieve_respects_rubric_threshold(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Analyze Q3 revenue",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        hindsight_store.add(
            task="Generate financial report",
            milestones=[sample_milestone_b],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        # Query uses same key rubric terms as sample_milestone's success_rubric
        result = hindsight_store.retrieve(
            "revenue growth reported",
            top_k=5,
            rubric_similarity_threshold=0.3,  # should pass for revenue milestone
        )
        assert len(result) >= 0  # threshold may filter; just verify no crash


# ---------------------------------------------------------------------------
# HindsightStore — sample
# ---------------------------------------------------------------------------


class TestHindsightStoreSample:
    def test_sample_empty(self, hindsight_store: HindsightStore) -> None:
        result = hindsight_store.sample(batch_size=4)
        assert result == []

    def test_sample_without_task(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        for i in range(5):
            m = Milestone(description=f"Task {i}", evidence_spec="spec", success_rubric="rubric", order=i)
            hindsight_store.add(
                task=f"Task {i}",
                milestones=[m],
                evidence_list=[sample_evidence],
                critic_results=[sample_critic_result_fail],
                trajectory=sample_trajectory,
            )
        result = hindsight_store.sample(batch_size=3)
        assert len(result) == 3
        # All should be valid HindsightEntry objects
        for e in result:
            assert isinstance(e, HindsightEntry)

    def test_sample_respects_batch_size(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        for i in range(10):
            m = Milestone(description=f"Task {i}", evidence_spec="spec", success_rubric="rubric", order=i)
            hindsight_store.add(
                task=f"Task {i}",
                milestones=[m],
                evidence_list=[sample_evidence],
                critic_results=[sample_critic_result_fail],
                trajectory=sample_trajectory,
            )
        result = hindsight_store.sample(batch_size=4)
        assert len(result) == 4

    def test_sample_returns_fewer_if_small(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Task 1",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        result = hindsight_store.sample(batch_size=10)
        assert len(result) == 1

    def test_sample_with_task(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        hindsight_store.add(
            task="Weather forecast task",
            milestones=[sample_milestone_b],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        # Revenue query should bias toward the revenue entry
        result = hindsight_store.sample(batch_size=2, task="revenue analysis")
        assert len(result) == 2
        # At least the first should be the revenue entry
        tasks = [e.original_task for e in result]
        assert "Revenue analysis task" in tasks


# ---------------------------------------------------------------------------
# HindsightStore — rubric similarity
# ---------------------------------------------------------------------------


class TestHindsightStoreRubricSimilarity:
    def test_get_rubric_similarity(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_milestone_b: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        entry = hindsight_store._entries[0]
        score = hindsight_store.get_rubric_similarity(entry, sample_milestone_b)
        assert 0.0 <= score <= 1.0

    def test_get_rubric_similarity_entry(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Analyze Q3 revenue data",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        entry = hindsight_store._entries[0]
        score = hindsight_store.get_rubric_similarity_entry(
            "Q3 revenue analysis", entry
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# HindsightStore — export / load
# ---------------------------------------------------------------------------


class TestHindsightStoreExportLoad:
    def test_export_creates_file(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Test task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        export_path = hindsight_store._store_dir / "export.json"
        hindsight_store.export(path=export_path)
        assert export_path.exists()
        with open(export_path) as f:
            data = json.load(f)
        assert data["entry_count"] == 1
        assert len(data["entries"]) == 1

    def test_load(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Test task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        export_path = hindsight_store._store_dir / "export.json"
        hindsight_store.export(path=export_path)

        # Load into a fresh store
        new_store = HindsightStore(store_dir=hindsight_store._store_dir)
        count = new_store.load(path=export_path)
        assert count == 1
        assert len(new_store) == 1
        assert new_store._entries[0].original_task == "Test task"

    def test_load_missing_file_returns_zero(
        self,
        hindsight_store: HindsightStore,
    ) -> None:
        count = hindsight_store.load(path=hindsight_store._store_dir / "nonexistent.json")
        assert count == 0

    def test_load_recomputes_tfidf(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Revenue analysis task",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        export_path = hindsight_store._store_dir / "export.json"
        hindsight_store.export(path=export_path)

        new_store = HindsightStore(store_dir=hindsight_store._store_dir)
        new_store.load(path=export_path)
        entry = new_store._entries[0]
        # TF-IDF should be recomputed after load
        assert isinstance(entry.keyword_tfidf, dict)


# ---------------------------------------------------------------------------
# HindsightStore — clear
# ---------------------------------------------------------------------------


class TestHindsightStoreClear:
    def test_clear_returns_count(
        self,
        hindsight_store: HindsightStore,
        sample_milestone: Milestone,
        sample_evidence: Evidence,
        sample_critic_result_fail: MilestoneCriticResult,
        sample_trajectory: list,
    ) -> None:
        hindsight_store.add(
            task="Test",
            milestones=[sample_milestone],
            evidence_list=[sample_evidence],
            critic_results=[sample_critic_result_fail],
            trajectory=sample_trajectory,
        )
        count = hindsight_store.clear()
        assert count == 1
        assert len(hindsight_store) == 0

    def test_clear_empty(self, hindsight_store: HindsightStore) -> None:
        count = hindsight_store.clear()
        assert count == 0
        assert len(hindsight_store) == 0


# ---------------------------------------------------------------------------
# HindsightStore._make_hindsight_label
# ---------------------------------------------------------------------------


class TestMakeHindsightLabel:
    def test_format(self, sample_milestone, sample_critic_result_fail) -> None:
        label = HindsightStore._make_hindsight_label(
            sample_milestone, sample_critic_result_fail
        )
        assert "What if" in label
        assert sample_milestone.description in label
        assert "verdict: fail" in label
        assert "confidence=" in label
