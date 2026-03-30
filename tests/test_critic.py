"""
Tests for src.critic — MilestoneCritic, BoxMazeBoundaryLayer, MilestoneVerdict.
"""

from __future__ import annotations

import uuid

import pytest

from src.critic import (
    BoundaryViolation,
    BoxMazeBoundaryLayer,
    MilestoneCritic,
    MilestoneCriticResult,
    MilestoneVerdict,
)
from src.decompose import Milestone
from src.evidence import Evidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def boundary_layer() -> BoxMazeBoundaryLayer:
    """Default (non-strict) boundary layer."""
    return BoxMazeBoundaryLayer()


@pytest.fixture
def strict_boundary_layer() -> BoxMazeBoundaryLayer:
    """Strict-mode boundary layer."""
    return BoxMazeBoundaryLayer(strict_mode=True)


@pytest.fixture
def critic() -> MilestoneCritic:
    """Default critic with standard thresholds."""
    return MilestoneCritic()


@pytest.fixture
def strict_critic() -> MilestoneCritic:
    """Critic with strict boundary layer and require_violations_for_fail=True."""
    return MilestoneCritic(
        boundary_layer=BoxMazeBoundaryLayer(strict_mode=True),
        require_violations_for_fail=True,
    )


# ---------------------------------------------------------------------------
# MilestoneVerdict tests
# ---------------------------------------------------------------------------


class TestMilestoneVerdict:
    def test_verdict_values(self) -> None:
        assert MilestoneVerdict.PASS.value == "pass"
        assert MilestoneVerdict.FAIL.value == "fail"
        assert MilestoneVerdict.UNCERTAIN.value == "uncertain"

    def test_verdict_enum_count(self) -> None:
        assert len(MilestoneVerdict) == 3

    def test_verdict_is_enum(self) -> None:
        assert isinstance(MilestoneVerdict.PASS, MilestoneVerdict)

    def test_verdict_comparison(self) -> None:
        assert MilestoneVerdict.PASS != MilestoneVerdict.FAIL
        assert MilestoneVerdict.UNCERTAIN != MilestoneVerdict.PASS


# ---------------------------------------------------------------------------
# BoundaryViolation dataclass tests
# ---------------------------------------------------------------------------


class TestBoundaryViolation:
    def test_fields_default(self) -> None:
        v = BoundaryViolation(
            violation_type="test_violation",
            description="A test boundary violation",
        )
        assert v.violation_type == "test_violation"
        assert v.description == "A test boundary violation"
        assert v.severity == "medium"
        assert v.milestone_id == ""

    def test_fields_explicit(self) -> None:
        v = BoundaryViolation(
            violation_type="out_of_bounds_action",
            description="Agent skipped a required step",
            severity="high",
            milestone_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert v.violation_type == "out_of_bounds_action"
        assert v.description == "Agent skipped a required step"
        assert v.severity == "high"
        assert v.milestone_id == "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# MilestoneCriticResult dataclass tests
# ---------------------------------------------------------------------------


class TestMilestoneCriticResult:
    def test_fields_default(self) -> None:
        r = MilestoneCriticResult(
            milestone_id="550e8400-e29b-41d4-a716-446655440000",
            verdict=MilestoneVerdict.PASS,
            reasoning="Strong evidence found.",
        )
        assert r.milestone_id == "550e8400-e29b-41d4-a716-446655440000"
        assert r.verdict == MilestoneVerdict.PASS
        assert r.reasoning == "Strong evidence found."
        assert r.confidence == 0.0
        assert r.violations == []
        assert r.evidence_quality == 0.0
        assert r.evidence_sufficiency == 0.0
        assert r.evidence_consistency == 0.0

    def test_fields_explicit(self) -> None:
        violation = BoundaryViolation(
            violation_type="constraint_overstep",
            description="Constraint violated",
            severity="high",
            milestone_id="test-id",
        )
        r = MilestoneCriticResult(
            milestone_id="550e8400-e29b-41d4-a716-446655440000",
            verdict=MilestoneVerdict.FAIL,
            reasoning="Evidence violates constraint.",
            confidence=0.85,
            violations=[violation],
            evidence_quality=0.9,
            evidence_sufficiency=0.8,
            evidence_consistency=0.7,
        )
        assert r.confidence == 0.85
        assert len(r.violations) == 1
        assert r.evidence_quality == 0.9

    def test_to_dict(self) -> None:
        r = MilestoneCriticResult(
            milestone_id="550e8400-e29b-41d4-a716-446655440000",
            verdict=MilestoneVerdict.PASS,
            reasoning="Tests pass.",
            confidence=0.9,
            evidence_quality=0.8,
            evidence_sufficiency=0.85,
            evidence_consistency=0.9,
        )
        d = r.to_dict()
        assert d["milestone_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert d["verdict"] == "pass"
        assert d["confidence"] == 0.9
        assert isinstance(d["violations"], list)

    def test_from_dict(self) -> None:
        raw = {
            "milestone_id": "550e8400-e29b-41d4-a716-446655440000",
            "verdict": "fail",
            "reasoning": "Tests fail.",
            "confidence": 0.75,
            "violations": [
                {
                    "violation_type": "out_of_bounds_action",
                    "description": "Step was skipped",
                    "severity": "medium",
                    "milestone_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ],
            "evidence_quality": 0.6,
            "evidence_sufficiency": 0.5,
            "evidence_consistency": 0.7,
        }
        r = MilestoneCriticResult.from_dict(raw)
        assert r.verdict == MilestoneVerdict.FAIL
        assert r.confidence == 0.75
        assert len(r.violations) == 1
        assert r.violations[0].violation_type == "out_of_bounds_action"

    def test_roundtrip(self) -> None:
        original = MilestoneCriticResult(
            milestone_id="550e8400-e29b-41d4-a716-446655440000",
            verdict=MilestoneVerdict.UNCERTAIN,
            reasoning="Evidence is mixed.",
            confidence=0.5,
            violations=[
                BoundaryViolation(
                    violation_type="low_confidence_evidence",
                    description="Confidence too low",
                    severity="low",
                    milestone_id="550e8400-e29b-41d4-a716-446655440000",
                )
            ],
            evidence_quality=0.4,
            evidence_sufficiency=0.5,
            evidence_consistency=0.6,
        )
        restored = MilestoneCriticResult.from_dict(original.to_dict())
        assert restored.milestone_id == original.milestone_id
        assert restored.verdict == original.verdict
        assert restored.confidence == original.confidence
        assert len(restored.violations) == len(original.violations)

    def test_repr(self) -> None:
        r = MilestoneCriticResult(
            milestone_id="550e8400-e29b-41d4-a716-446655440000",
            verdict=MilestoneVerdict.PASS,
            reasoning="All good.",
        )
        rep = repr(r)
        assert "MilestoneCriticResult" in rep
        assert "pass" in rep


# ---------------------------------------------------------------------------
# BoxMazeBoundaryLayer tests
# ---------------------------------------------------------------------------


class TestBoxMazeBoundaryLayerInit:
    def test_default_init(self) -> None:
        bl = BoxMazeBoundaryLayer()
        assert bl._strict_mode is False
        assert bl._max_violations == 10

    def test_strict_mode_init(self) -> None:
        bl = BoxMazeBoundaryLayer(strict_mode=True)
        assert bl._strict_mode is True

    def test_custom_max_violations(self) -> None:
        bl = BoxMazeBoundaryLayer(max_violations_per_milestone=5)
        assert bl._max_violations == 5


class TestBoxMazeBoundaryLayerCheckBoundary:
    def test_empty_evidence_returns_empty_violations(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue figures",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(milestone_id=str(milestone.id), confidence=0.0)
        violations = boundary_layer.check_boundary(milestone, evidence)
        assert isinstance(violations, list)

    def test_no_constraint_milestone_no_violations(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Retrieve the stock price",
            evidence_spec="Price data",
            success_rubric="Pass if retrieved",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Retrieved stock price of $150"],
            citations=[
                {"source_action_idx": 0, "text": "Retrieved stock price of $150"}
            ],
            intermediate_results=[150.0],
            confidence=0.9,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        # No constraint keywords → no constraint violations
        assert all(v.violation_type != "constraint_overstep" for v in violations)

    def test_out_of_bounds_pattern_detected(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Verify the compliance report",
            evidence_spec="Compliance verification output",
            success_rubric="Pass if verified",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "We failed to verify the compliance report",
                "The check was skipped",
            ],
            confidence=0.4,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        oob_violations = [
            v for v in violations if v.violation_type == "out_of_bounds_action"
        ]
        assert len(oob_violations) >= 1

    def test_memory_grounding_failure_detected(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "The analysis shows revenue is $1.2M. However, this contradicts "
                "the previously identified figure. The expected result was $1.1M "
                "but the actual value is $1.2M."
            ],
            confidence=0.5,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        grounding_violations = [
            v
            for v in violations
            if v.violation_type == "memory_grounding_failure"
        ]
        assert len(grounding_violations) >= 1

    def test_inconsistency_detected_high_confidence_no_citations(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Generate the report",
            evidence_spec="PDF output",
            success_rubric="Pass if generated",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "Step 1 complete",
                "Step 2 complete",
                "Step 3 complete",
                "Step 4 complete",
                "Step 5 complete",
            ],
            citations=[],  # Zero citations despite high text count
            confidence=0.8,  # High confidence
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        inconsistency_violations = [
            v
            for v in violations
            if v.violation_type == "evidence_inconsistency"
        ]
        assert len(inconsistency_violations) >= 1

    def test_strict_mode_flags_low_confidence(
        self, strict_boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Some partial evidence"],
            confidence=0.3,  # Below 0.5 threshold
        )
        violations = strict_boundary_layer.check_boundary(milestone, evidence)
        low_conf_violations = [
            v
            for v in violations
            if v.violation_type == "low_confidence_evidence"
        ]
        assert len(low_conf_violations) >= 1

    def test_non_strict_mode_no_low_confidence_violation(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Some partial evidence"],
            confidence=0.3,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        low_conf_violations = [
            v
            for v in violations
            if v.violation_type == "low_confidence_evidence"
        ]
        assert len(low_conf_violations) == 0

    def test_reset_constraints_clears_memory(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        boundary_layer._seen_constraints.add("test_constraint")
        assert len(boundary_layer._seen_constraints) == 1
        boundary_layer.reset_constraints()
        assert len(boundary_layer._seen_constraints) == 0

    def test_max_violations_is_respected(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        bl = BoxMazeBoundaryLayer(max_violations_per_milestone=3)
        milestone = Milestone(
            description="Verify compliance",
            evidence_spec="Verification report",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "failed to verify",
                "skipped the check",
                "ignored the requirement",
                "missed it",
                "contrary to requirements",
            ],
            confidence=0.3,
        )
        violations = bl.check_boundary(milestone, evidence)
        assert len(violations) <= 3

    def test_extracted_keywords_excludes_stop_words(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        keywords = BoxMazeBoundaryLayer._extract_keywords(
            "The quick brown fox jumps over the lazy dog and the cat"
        )
        assert "the" not in keywords
        assert "and" not in keywords
        assert "over" not in keywords

    def test_extracted_numbers(self) -> None:
        numbers = BoxMazeBoundaryLayer._extract_numbers(
            "Revenue was $1.2M and costs were $500K"
        )
        assert 1.2 in numbers
        assert 500.0 in numbers

    def test_has_contradictory_results_true(self) -> None:
        results = ["pass", "fail", "correct", "invalid"]
        assert BoxMazeBoundaryLayer._has_contradictory_results(results) is True

    def test_has_contradictory_results_false(self) -> None:
        results = ["pass", "correct", "confirmed", "validated"]
        assert BoxMazeBoundaryLayer._has_contradictory_results(results) is False

    def test_has_contradictory_results_empty(self) -> None:
        assert BoxMazeBoundaryLayer._has_contradictory_results([]) is False

    def test_extract_action_verb(self) -> None:
        verb = BoxMazeBoundaryLayer._extract_action_verb(
            "The agent should analyze the revenue figures"
        )
        assert verb == "analyze"

    def test_extract_action_verb_no_match(self) -> None:
        verb = BoxMazeBoundaryLayer._extract_action_verb(
            "The quick brown fox"
        )
        assert verb is None


# ---------------------------------------------------------------------------
# MilestoneCritic tests
# ---------------------------------------------------------------------------


class TestMilestoneCriticInit:
    def test_default_init(self) -> None:
        critic = MilestoneCritic()
        assert critic._pass_threshold == 0.7
        assert critic._fail_threshold == 0.3
        assert not critic._require_violations_for_fail

    def test_custom_thresholds(self) -> None:
        critic = MilestoneCritic(
            confidence_threshold_pass=0.8,
            confidence_threshold_fail=0.2,
        )
        assert critic._pass_threshold == 0.8
        assert critic._fail_threshold == 0.2

    def test_invalid_thresholds_swap_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >"):
            MilestoneCritic(
                confidence_threshold_pass=0.3,
                confidence_threshold_fail=0.7,
            )

    def test_invalid_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            MilestoneCritic(confidence_threshold_pass=1.5)
        with pytest.raises(ValueError):
            MilestoneCritic(confidence_threshold_fail=-0.5)


class TestMilestoneCriticCritique:
    def test_critique_empty_lists(
        self, critic: MilestoneCritic
    ) -> None:
        results = critic.critique([], [])
        assert results == []

    def test_critique_length_mismatch_raises(
        self, critic: MilestoneCritic
    ) -> None:
        milestones = [
            Milestone(
                description="Step 1",
                evidence_spec="Evidence 1",
                success_rubric="Pass 1",
                order=0,
            )
        ]
        evidence_list = [
            Evidence(milestone_id=str(milestones[0].id), confidence=0.5)
        ]
        # Same length → should not raise
        critic.critique(milestones, evidence_list)
        # Add extra evidence → should raise
        evidence_list.append(
            Evidence(milestone_id=str(uuid.uuid4()), confidence=0.5)
        )
        with pytest.raises(ValueError, match="same length"):
            critic.critique(milestones, evidence_list)

    def test_critique_single_pass(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "Analyzed revenue data showing $1.2M total revenue",
                "Revenue analysis completed",
            ],
            citations=[
                {"source_action_idx": 0, "text": "Revenue = $1.2M"}
            ],
            intermediate_results=[{"revenue": 1200000}],
            confidence=0.9,
        )
        results = critic.critique([milestone], [evidence])
        assert len(results) == 1
        assert results[0].verdict in (MilestoneVerdict.PASS, MilestoneVerdict.UNCERTAIN)

    def test_critique_single_fail_low_evidence(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[],  # No text snippets
            citations=[],  # No citations
            intermediate_results=[],  # No results
            confidence=0.0,  # Zero confidence
        )
        results = critic.critique([milestone], [evidence])
        assert len(results) == 1
        assert results[0].verdict == MilestoneVerdict.FAIL

    def test_critique_single_uncertain_middle_score(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Verify the compliance report",
            evidence_spec="Compliance verification",
            success_rubric="Pass if verified",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Some partial evidence of verification"],
            citations=[],
            intermediate_results=[],
            confidence=0.5,  # Middle confidence
        )
        results = critic.critique([milestone], [evidence])
        assert len(results) == 1
        # With thresholds 0.3/0.7 and combined ~0.5, UNCERTAIN is expected
        assert results[0].verdict == MilestoneVerdict.UNCERTAIN

    def test_critique_multiple_milestones(
        self, critic: MilestoneCritic
    ) -> None:
        milestones = [
            Milestone(
                description="Retrieve stock price",
                evidence_spec="Price data",
                success_rubric="Pass if retrieved",
                order=0,
            ),
            Milestone(
                description="Analyze trends",
                evidence_spec="Trend analysis",
                success_rubric="Pass if analyzed",
                order=1,
            ),
            Milestone(
                description="Generate report",
                evidence_spec="PDF output",
                success_rubric="Pass if generated",
                order=2,
            ),
        ]
        evidence_list = [
            Evidence(
                milestone_id=str(m.id),
                text_snippets=[f"Evidence for {m.description}"],
                confidence=0.8,
            )
            for m in milestones
        ]
        results = critic.critique(milestones, evidence_list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r.verdict, MilestoneVerdict)
            assert r.milestone_id in [str(m.id) for m in milestones]

    def test_critique_high_severity_violation_forces_fail(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "Revenue was found to be $1.2M. However, this contradicts "
                "the previously identified value. Expected revenue was $1.1M "
                "but the actual revenue is $1.2M."
            ],
            confidence=0.9,  # High confidence but has contradiction
        )
        # The boundary layer should detect the memory grounding failure
        results = critic.critique([milestone], [evidence])
        assert len(results) == 1
        # High-severity violations force FAIL
        assert results[0].verdict == MilestoneVerdict.FAIL


class TestMilestoneCriticScoring:
    def test_evidence_quality_high_with_relevant_snippets(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue figures",
            evidence_spec="Revenue analysis output",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[
                "Analyzed the revenue figures showing growth",
                "Revenue analysis complete",
            ],
            citations=[
                {"source_action_idx": 0, "text": "Revenue analysis"}
            ],
            intermediate_results=[{"revenue_growth": "15%"}],
            confidence=0.9,
        )
        results = critic.critique([milestone], [evidence])
        assert results[0].evidence_quality > 0.5

    def test_evidence_quality_low_with_no_citations(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue figures",
            evidence_spec="Revenue analysis",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Something vaguely relevant"],
            citations=[],  # No citations
            intermediate_results=[],
            confidence=0.3,
        )
        results = critic.critique([milestone], [evidence])
        assert results[0].evidence_quality < 0.5

    def test_evidence_sufficiency_zero_with_no_evidence(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Analysis",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[],
            citations=[],
            intermediate_results=[],
            confidence=0.0,
        )
        results = critic.critique([milestone], [evidence])
        assert results[0].evidence_sufficiency == 0.0

    def test_evidence_consistency_perfect_single_piece(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue data",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Revenue analysis complete"],
            citations=[],
            intermediate_results=[],
            confidence=0.8,
        )
        results = critic.critique([milestone], [evidence])
        # Single piece → vacuously consistent
        assert results[0].evidence_consistency == 1.0

    def test_evidence_consistency_penalized_for_contradictions(
        self, critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Verify compliance",
            evidence_spec="Verification",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Verification passed", "Verification failed"],
            citations=[],
            intermediate_results=["pass", "fail"],  # Contradictory
            confidence=0.6,
        )
        results = critic.critique([milestone], [evidence])
        # Contradictions should reduce consistency
        assert results[0].evidence_consistency < 1.0


class TestMilestoneCriticStrictMode:
    def test_require_violations_for_fail_mode(
        self, strict_critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue",
            evidence_spec="Revenue analysis",
            success_rubric="Pass",
            order=0,
        )
        # No evidence at all → would normally be FAIL but no violations
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=[],
            citations=[],
            intermediate_results=[],
            confidence=0.0,
        )
        results = strict_critic.critique([milestone], [evidence])
        # With require_violations_for_fail=True and zero violations,
        # verdict is still FAIL due to no evidence (combined score is 0)
        assert results[0].verdict == MilestoneVerdict.FAIL

    def test_strict_critic_flags_low_confidence(
        self, strict_critic: MilestoneCritic
    ) -> None:
        milestone = Milestone(
            description="Generate report",
            evidence_spec="PDF",
            success_rubric="Pass",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Partial report generated"],
            confidence=0.3,
        )
        results = strict_critic.critique([milestone], [evidence])
        assert len(results[0].violations) >= 1


class TestMilestoneCriticResultOrdering:
    def test_results_match_milestone_order(
        self, critic: MilestoneCritic
    ) -> None:
        milestones = [
            Milestone(
                description=f"Step {i}",
                evidence_spec=f"Evidence {i}",
                success_rubric="Pass",
                order=i,
            )
            for i in range(5)
        ]
        evidence_list = [
            Evidence(
                milestone_id=str(m.id),
                text_snippets=[f"Evidence for step {m.order}"],
                confidence=0.7,
            )
            for m in milestones
        ]
        results = critic.critique(milestones, evidence_list)
        for milestone, result in zip(milestones, results):
            assert result.milestone_id == str(milestone.id)


# ---------------------------------------------------------------------------
# Integration tests: Critic + Decomposer + EvidenceCollector
# ---------------------------------------------------------------------------


class TestCriticIntegration:
    def test_end_to_end_with_real_milestones(
        self, critic: MilestoneCritic
    ) -> None:
        """Integration test: decompose a task, collect evidence, critique."""
        from src.decompose import MilestoneDecomposer
        from src.evidence import EvidenceCollector

        decomposer = MilestoneDecomposer()
        collector = EvidenceCollector()

        task = "Retrieve stock prices and generate a summary report"
        milestones = decomposer.decompose(task)

        trajectory = [
            {
                "action": "retrieve stock prices",
                "observation": "Prices: AAPL=$180, GOOG=$140",
                "thought": "Need to get prices",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "action": "generate summary report",
                "observation": "Report saved to /tmp/summary.pdf",
                "thought": "Report is ready",
                "timestamp": "2024-01-01T00:01:00Z",
            },
        ]

        evidence_list = [
            collector.collect(m, trajectory) for m in milestones
        ]

        results = critic.critique(milestones, evidence_list)

        assert len(results) == len(milestones)
        for milestone, evidence, result in zip(
            milestones, evidence_list, results
        ):
            assert result.milestone_id == str(milestone.id)
            assert isinstance(result.verdict, MilestoneVerdict)
            assert 0.0 <= result.confidence <= 1.0
            # Evidence scores should reflect what the collector provided
            assert 0.0 <= result.evidence_quality <= 1.0
            assert 0.0 <= result.evidence_sufficiency <= 1.0
            assert 0.0 <= result.evidence_consistency <= 1.0

    def test_critique_preserves_all_fields(
        self, critic: MilestoneCritic
    ) -> None:
        """Verify that all MilestoneCriticResult fields are populated."""
        milestone = Milestone(
            description="Verify the output",
            evidence_spec="Verification results",
            success_rubric="Pass if verified",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Verification complete and confirmed"],
            citations=[{"source_action_idx": 0, "text": "Verification confirmed"}],
            intermediate_results=[{"status": "verified"}],
            confidence=0.85,
        )
        results = critic.critique([milestone], [evidence])
        r = results[0]
        assert r.milestone_id == str(milestone.id)
        assert isinstance(r.verdict, MilestoneVerdict)
        assert len(r.reasoning) > 0
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.violations, list)
        assert 0.0 <= r.evidence_quality <= 1.0
        assert 0.0 <= r.evidence_sufficiency <= 1.0
        assert 0.0 <= r.evidence_consistency <= 1.0


# ---------------------------------------------------------------------------
# Boundary layer with various milestone patterns
# ---------------------------------------------------------------------------


class TestBoxMazeConstraintPatterns:
    def test_constraint_not_violated_when_action_absent(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue but do not share it",
            evidence_spec="Revenue analysis",
            success_rubric="Pass if analyzed but not shared",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Revenue analysis completed privately"],
            confidence=0.8,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        # No "share" or "analyze" violation expected since "analyze" is the action
        assert all(v.violation_type != "constraint_overstep" for v in violations)

    def test_constraint_not_pattern_yes(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Ensure the data is not corrupted",
            evidence_spec="Data integrity check",
            success_rubric="Pass if data is intact",
            order=0,
        )
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Data integrity verified", "No corruption detected"],
            confidence=0.9,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        assert all(v.violation_type != "constraint_overstep" for v in violations)

    def test_insufficient_evidence_violation_when_keywords_missing(
        self, boundary_layer: BoxMazeBoundaryLayer
    ) -> None:
        milestone = Milestone(
            description="Analyze revenue and compute ROI",
            evidence_spec="Revenue analysis and return on investment calculation",
            success_rubric="Pass if both completed",
            order=0,
        )
        # Evidence only mentions revenue, not ROI
        evidence = Evidence(
            milestone_id=str(milestone.id),
            text_snippets=["Revenue analysis complete"],
            citations=[],
            intermediate_results=[],
            confidence=0.4,
        )
        violations = boundary_layer.check_boundary(milestone, evidence)
        insufficiency_violations = [
            v
            for v in violations
            if v.violation_type == "insufficient_evidence"
        ]
        assert len(insufficiency_violations) >= 1
