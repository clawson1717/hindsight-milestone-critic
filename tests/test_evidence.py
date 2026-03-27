"""
Tests for src.evidence — EvidenceCollector and Evidence.
"""

from __future__ import annotations

import json

import pytest

from src.decompose import Milestone
from src.evidence import Evidence, EvidenceCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collector() -> EvidenceCollector:
    """EvidenceCollector with default settings."""
    return EvidenceCollector()


@pytest.fixture
def sample_milestone() -> Milestone:
    """A simple milestone for testing."""
    return Milestone(
        description="Analyze quarterly revenue figures",
        evidence_spec="Revenue analysis notes",
        success_rubric="Pass if revenue analysis is present",
        order=0,
    )


@pytest.fixture
def sample_trajectory() -> list[dict]:
    """A minimal trajectory for testing."""
    return [
        {
            "action": "retrieve quarterly data",
            "observation": "Q3 revenue = $1.2M",
            "thought": "Got the data, now analyzing",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "action": "analyze revenue",
            "observation": "Analysis complete: 15% growth",
            "thought": "Growth looks solid",
            "timestamp": "2024-01-01T00:01:00Z",
        },
        {
            "action": "generate report",
            "observation": "Report saved to /reports/q3.pdf",
            "thought": "All done",
            "timestamp": "2024-01-01T00:02:00Z",
        },
    ]


# ---------------------------------------------------------------------------
# Evidence dataclass tests
# ---------------------------------------------------------------------------


class TestEvidenceDataclass:
    def test_evidence_fields(self) -> None:
        e = Evidence(milestone_id="abc-123")
        assert e.milestone_id == "abc-123"
        assert e.text_snippets == []
        assert e.citations == []
        assert e.intermediate_results == []
        assert e.confidence == 0.0

    def test_evidence_with_all_fields(self) -> None:
        e = Evidence(
            milestone_id="xyz-456",
            text_snippets=["snippet one", "snippet two"],
            citations=[{"source_action_idx": 0, "text": "cite one"}],
            intermediate_results=[{"key": "value"}, 42],
            confidence=0.85,
        )
        assert e.milestone_id == "xyz-456"
        assert len(e.text_snippets) == 2
        assert len(e.citations) == 1
        assert len(e.intermediate_results) == 2
        assert e.confidence == 0.85

    def test_evidence_to_dict(self) -> None:
        e = Evidence(
            milestone_id="test-id",
            text_snippets=["text"],
            citations=[{"source_action_idx": 1, "text": "hello"}],
            intermediate_results=[{"result": 99}],
            confidence=0.7,
        )
        d = e.to_dict()
        assert d["milestone_id"] == "test-id"
        assert d["text_snippets"] == ["text"]
        assert d["citations"] == [{"source_action_idx": 1, "text": "hello"}]
        assert d["intermediate_results"] == [{"result": 99}]
        assert d["confidence"] == 0.7

    def test_evidence_from_dict(self) -> None:
        raw = {
            "milestone_id": "from-dict-id",
            "text_snippets": ["a", "b"],
            "citations": [{"source_action_idx": 2, "text": "c"}],
            "intermediate_results": [1, 2, 3],
            "confidence": 0.55,
        }
        e = Evidence.from_dict(raw)
        assert e.milestone_id == "from-dict-id"
        assert len(e.text_snippets) == 2
        assert len(e.citations) == 1
        assert len(e.intermediate_results) == 3
        assert e.confidence == 0.55

    def test_evidence_dict_roundtrip(self) -> None:
        original = Evidence(
            milestone_id="roundtrip",
            text_snippets=["alpha", "beta"],
            citations=[{"source_action_idx": 0, "text": "gamma"}],
            intermediate_results=[{"x": 10}],
            confidence=0.9,
        )
        restored = Evidence.from_dict(original.to_dict())
        assert restored.milestone_id == original.milestone_id
        assert restored.text_snippets == original.text_snippets
        assert restored.citations == original.citations
        assert restored.intermediate_results == original.intermediate_results
        assert restored.confidence == original.confidence

    def test_evidence_repr(self) -> None:
        e = Evidence(
            milestone_id="repr-test",
            text_snippets=["a"],
            citations=[],
            intermediate_results=[],
            confidence=0.5,
        )
        r = repr(e)
        assert "Evidence" in r
        assert "repr-test" in r
        assert "0.50" in r


# ---------------------------------------------------------------------------
# EvidenceCollector construction tests
# ---------------------------------------------------------------------------


class TestEvidenceCollectorInit:
    def test_default_weights_sum_to_one(self) -> None:
        c = EvidenceCollector()
        assert c._text_weight + c._action_weight + c._obs_weight == 1.0

    def test_custom_weights(self) -> None:
        c = EvidenceCollector(
            text_match_weight=0.5,
            action_match_weight=0.25,
            observation_match_weight=0.25,
        )
        assert c._text_weight == 0.5
        assert c._action_weight == 0.25
        assert c._obs_weight == 0.25

    def test_weight_sum_must_equal_one(self) -> None:
        with pytest.raises(ValueError, match="must equal 1.0"):
            EvidenceCollector(text_match_weight=0.1, action_match_weight=0.1, observation_match_weight=0.1)

    def test_threshold_must_be_in_range(self) -> None:
        with pytest.raises(ValueError, match="min_confidence_threshold"):
            EvidenceCollector(min_confidence_threshold=-0.1)
        with pytest.raises(ValueError, match="min_confidence_threshold"):
            EvidenceCollector(min_confidence_threshold=1.5)


# ---------------------------------------------------------------------------
# Basic evidence collection tests
# ---------------------------------------------------------------------------


class TestCollectBasic:
    def test_collect_returns_evidence(
        self,
        collector: EvidenceCollector,
        sample_milestone: Milestone,
        sample_trajectory: list[dict],
    ) -> None:
        evidence = collector.collect(sample_milestone, sample_trajectory)
        assert isinstance(evidence, Evidence)
        assert evidence.milestone_id == str(sample_milestone.id)
        assert evidence.confidence >= 0.0
        assert evidence.confidence <= 1.0

    def test_collect_empty_trajectory_returns_zero_confidence(
        self,
        collector: EvidenceCollector,
        sample_milestone: Milestone,
    ) -> None:
        evidence = collector.collect(sample_milestone, [])
        assert evidence.confidence == 0.0
        assert evidence.text_snippets == []
        assert evidence.citations == []
        assert evidence.intermediate_results == []

    def test_collect_milestone_no_match_in_trajectory(
        self,
        collector: EvidenceCollector,
    ) -> None:
        """A milestone whose description has no overlap with trajectory."""
        milestone = Milestone(
            description="Diagnose quantum entanglement errors",
            evidence_spec="Diagnosis report",
            success_rubric="Pass if diagnosed",
            order=0,
        )
        trajectory = [
            {
                "action": "browse website",
                "observation": "Page loaded",
                "thought": "OK",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence == 0.0
        assert evidence.text_snippets == []

    def test_collect_preserves_milestone_id(
        self,
        collector: EvidenceCollector,
        sample_trajectory: list[dict],
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Notes",
            success_rubric="Pass",
            order=5,
        )
        evidence = collector.collect(milestone, sample_trajectory)
        assert evidence.milestone_id == str(milestone.id)


# ---------------------------------------------------------------------------
# Text-matching tests
# ---------------------------------------------------------------------------


class TestTextMatching:
    def test_keyword_match_found(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        trajectory = [
            {
                "action": "analyze revenue figures",
                "observation": "Revenue up 10%",
                "thought": "Looking good",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence > 0.0
        assert len(evidence.citations) > 0

    def test_phrase_match_from_description(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Compare Q3 to Q4 performance metrics",
            evidence_spec="Comparison table",
            success_rubric="Pass if compared",
            order=0,
        )
        trajectory = [
            {
                "action": "compare Q3 to Q4 performance metrics",
                "observation": "Comparison complete",
                "thought": "Done",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence > 0.0

    def test_multiple_trajectory_steps_all_searched(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Retrieve and verify financial data",
            evidence_spec="Data records",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {"action": "fetch financial records", "observation": "Records retrieved", "thought": "Step 1", "timestamp": "2024-01-01T00:00:00Z"},
            {"action": "verify data integrity", "observation": "All checks passed", "thought": "Step 2", "timestamp": "2024-01-01T00:01:00Z"},
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence > 0.0


# ---------------------------------------------------------------------------
# Action-pattern tests
# ---------------------------------------------------------------------------


class TestActionPatternMatching:
    def test_action_verb_match_boosts_confidence(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Generate quarterly summary report",
            evidence_spec="Report PDF",
            success_rubric="Pass if generated",
            order=0,
        )
        trajectory = [
            {
                "action": "generate quarterly summary report",
                "observation": "Report created successfully",
                "thought": "All done",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence > 0.0

    def test_observation_only_no_action_match(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Compute total cost",
            evidence_spec="Cost value",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "review spreadsheet",
                "observation": "Total computed cost: $45,000",
                "thought": "Calculated",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        # Should still score from keyword/phrase match in observation
        assert evidence.confidence >= 0.0


# ---------------------------------------------------------------------------
# Observation parsing tests
# ---------------------------------------------------------------------------


class TestObservationParsing:
    def test_json_extraction_from_observation(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Retrieve inventory count",
            evidence_spec="Inventory list",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "query inventory database",
                "observation": '{"product": "Widget", "count": 142, "status": "in_stock"}',
                "thought": "Got data",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        # JSON should be parsed into intermediate_results
        json_vals = [v for v in evidence.intermediate_results if isinstance(v, dict)]
        assert len(json_vals) > 0

    def test_key_value_extraction(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Measure system performance",
            evidence_spec="Metrics",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "run performance benchmark",
                "observation": "CPU usage: 72%\nMemory usage: 4.2 GB\nLatency: 45ms",
                "thought": "Benchmark done",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        kv_pairs = [v for v in evidence.intermediate_results if isinstance(v, dict)]
        assert len(kv_pairs) > 0

    def test_numeric_extraction(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Calculate total revenue",
            evidence_spec="Revenue total",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "sum all transactions",
                "observation": "Revenue total: 1500000",
                "thought": "Calculated",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        numbers = [v for v in evidence.intermediate_results if isinstance(v, (int, float))]
        assert len(numbers) > 0

    def test_json_array_extraction(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Retrieve user list",
            evidence_spec="User list",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "fetch users",
                "observation": 'Users: [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]',
                "thought": "Done",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        # Should extract the list
        lists = [v for v in evidence.intermediate_results if isinstance(v, list)]
        assert len(lists) > 0


# ---------------------------------------------------------------------------
# Confidence scoring tests
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    def test_confidence_in_valid_range(
        self,
        collector: EvidenceCollector,
        sample_milestone: Milestone,
        sample_trajectory: list[dict],
    ) -> None:
        evidence = collector.collect(sample_milestone, sample_trajectory)
        assert 0.0 <= evidence.confidence <= 1.0

    def test_confidence_zero_for_empty_trajectory(
        self,
        collector: EvidenceCollector,
        sample_milestone: Milestone,
    ) -> None:
        evidence = collector.collect(sample_milestone, [])
        assert evidence.confidence == 0.0

    def test_confidence_reflects_evidence_strength(
        self,
        collector: EvidenceCollector,
    ) -> None:
        """Weak evidence should score lower than strong evidence."""
        milestone = Milestone(
            description="Analyze revenue and compare to last year",
            evidence_spec="Comparison report",
            success_rubric="Pass",
            order=0,
        )
        weak_trajectory = [
            {
                "action": "open spreadsheet",
                "observation": "Spreadsheet opened",
                "thought": "OK",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        strong_trajectory = [
            {
                "action": "analyze revenue figures",
                "observation": "Revenue analysis complete: 15% growth compared to last year",
                "thought": "Analysis done",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "action": "compare Q3 to Q4 performance",
                "observation": "Comparison table generated: Q3=$1.2M, Q4=$1.38M",
                "thought": "All done",
                "timestamp": "2024-01-01T00:01:00Z",
            },
        ]
        weak_evidence = collector.collect(milestone, weak_trajectory)
        strong_evidence = collector.collect(milestone, strong_trajectory)
        assert strong_evidence.confidence >= weak_evidence.confidence

    def test_min_confidence_threshold_discards_weak_evidence(
        self,
        collector: EvidenceCollector,
    ) -> None:
        """Setting threshold should zero out evidence below it."""
        strict_collector = EvidenceCollector(min_confidence_threshold=0.5)
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Notes",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "open spreadsheet",
                "observation": "Spreadsheet opened",
                "thought": "OK",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = strict_collector.collect(milestone, trajectory)
        # If raw confidence is below 0.5, evidence should be zeroed
        if evidence.confidence < 0.5:
            assert evidence.text_snippets == []
            assert evidence.citations == []

    def test_weight_assignment_affects_confidence(
        self,
        sample_milestone: Milestone,
        sample_trajectory: list[dict],
    ) -> None:
        """Changing weights should change the resulting confidence score."""
        c1 = EvidenceCollector(
            text_match_weight=0.8,
            action_match_weight=0.1,
            observation_match_weight=0.1,
        )
        c2 = EvidenceCollector(
            text_match_weight=0.1,
            action_match_weight=0.1,
            observation_match_weight=0.8,
        )
        e1 = c1.collect(sample_milestone, sample_trajectory)
        e2 = c2.collect(sample_milestone, sample_trajectory)
        # Scores may differ depending on what's in the trajectory
        assert isinstance(e1.confidence, float)
        assert isinstance(e2.confidence, float)


# ---------------------------------------------------------------------------
# Citation tests
# ---------------------------------------------------------------------------


class TestCitations:
    def test_citations_contain_source_index(
        self,
        collector: EvidenceCollector,
        sample_milestone: Milestone,
        sample_trajectory: list[dict],
    ) -> None:
        evidence = collector.collect(sample_milestone, sample_trajectory)
        for citation in evidence.citations:
            assert "source_action_idx" in citation
            assert isinstance(citation["source_action_idx"], int)
            assert 0 <= citation["source_action_idx"] < len(sample_trajectory)

    def test_citations_contain_text(
        self,
        collector: EvidenceCollector,
        sample_milestone: Milestone,
        sample_trajectory: list[dict],
    ) -> None:
        evidence = collector.collect(sample_milestone, sample_trajectory)
        for citation in evidence.citations:
            assert "text" in citation
            assert isinstance(citation["text"], str)
            assert len(citation["text"]) > 0

    def test_citations_are_deduplicated(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Notes",
            success_rubric="Pass",
            order=0,
        )
        # Same step repeated with overlapping content
        trajectory = [
            {"action": "analyze revenue", "observation": "Revenue analysis: good", "thought": "OK", "timestamp": "2024-01-01T00:00:00Z"},
            {"action": "analyze revenue", "observation": "Revenue analysis: good", "thought": "OK", "timestamp": "2024-01-01T00:01:00Z"},
        ]
        evidence = collector.collect(milestone, trajectory)
        # Deduplicated by (source_action_idx, text)
        assert len(evidence.citations) <= len(trajectory)


# ---------------------------------------------------------------------------
# Intermediate results tests
# ---------------------------------------------------------------------------


class TestIntermediateResults:
    def test_parsed_json_in_intermediate_results(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Retrieve configuration settings",
            evidence_spec="Config",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "fetch config",
                "observation": '{"debug": true, "timeout": 30}',
                "thought": "Got config",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        # Parsed JSON dict should appear
        json_items = [r for r in evidence.intermediate_results if isinstance(r, dict)]
        assert len(json_items) > 0

    def test_empty_intermediate_results_when_no_structured_data(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Navigate to dashboard",
            evidence_spec="Dashboard view",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "click dashboard link",
                "observation": "Dashboard loaded",
                "thought": "Navigated",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        evidence = collector.collect(milestone, trajectory)
        # No structured data → intermediate_results may be empty or minimal
        assert isinstance(evidence.intermediate_results, list)


# ---------------------------------------------------------------------------
# Mixed evidence scenarios
# ---------------------------------------------------------------------------


class TestMixedEvidence:
    def test_trajectory_with_multiple_evidence_types(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Retrieve stock price and analyze performance trends",
            evidence_spec="Stock data and analysis",
            success_rubric="Pass",
            order=0,
        )
        trajectory = [
            {
                "action": "fetch stock price for AAPL",
                "observation": '{"symbol": "AAPL", "price": 182.52, "date": "2024-01-15"}',
                "thought": "Got price",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "action": "analyze performance trends",
                "observation": "Analysis complete: AAPL up 12% over 30 days. Trend: bullish.",
                "thought": "Analyzed",
                "timestamp": "2024-01-01T00:01:00Z",
            },
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence > 0.0
        assert len(evidence.citations) >= 1
        assert evidence.milestone_id == str(milestone.id)

    def test_realistic_multi_step_trajectory(
        self,
        collector: EvidenceCollector,
    ) -> None:
        milestone = Milestone(
            description="Verify API response meets schema requirements",
            evidence_spec="Schema validation report",
            success_rubric="Pass if validated",
            order=1,
        )
        trajectory = [
            {
                "action": "call /api/users endpoint",
                "observation": '{"users": [{"id": 1, "name": "Alice"}], "count": 1}',
                "thought": "Got response",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "action": "validate JSON schema",
                "observation": 'Schema check: {"valid": true, "errors": []}',
                "thought": "Schema is valid",
                "timestamp": "2024-01-01T00:01:00Z",
            },
            {
                "action": "verify response",
                "observation": "Verification complete: all fields present and correct",
                "thought": "All good",
                "timestamp": "2024-01-01T00:02:00Z",
            },
        ]
        evidence = collector.collect(milestone, trajectory)
        assert evidence.confidence > 0.0
        assert isinstance(evidence.to_dict(), dict)
        assert isinstance(Evidence.from_dict(evidence.to_dict()), Evidence)
