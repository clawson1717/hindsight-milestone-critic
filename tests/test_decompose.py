"""
Tests for src.decompose — MilestoneDecomposer and Milestone.
"""

from __future__ import annotations

import uuid

import pytest

from src.decompose import (
    Milestone,
    MilestoneDecomposer,
    _clause_to_milestone,
    _identify_verb,
    _make_evidence_spec,
    _make_success_rubric,
    _split_into_clauses,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def decomposer() -> MilestoneDecomposer:
    """Decomposer with no API key (heuristic-only)."""
    return MilestoneDecomposer(openai_api_key=None)


@pytest.fixture
def decomposer_with_key() -> MilestoneDecomposer:
    """Decomposer with a dummy key (refine still won't hit the network in tests)."""
    return MilestoneDecomposer(openai_api_key="sk-test-dummy")


# ---------------------------------------------------------------------------
# Milestone dataclass tests
# ---------------------------------------------------------------------------


class TestMilestone:
    def test_milestone_fields(self) -> None:
        m = Milestone(
            description="Analyze Q3 revenue",
            evidence_spec="Q3 revenue figures",
            success_rubric="Pass if Q3 revenue is reported",
            order=0,
        )
        assert m.description == "Analyze Q3 revenue"
        assert m.evidence_spec == "Q3 revenue figures"
        assert m.success_rubric == "Pass if Q3 revenue is reported"
        assert m.order == 0
        assert isinstance(m.id, uuid.UUID)

    def test_milestone_to_dict(self) -> None:
        m = Milestone(
            description="Retrieve stock price",
            evidence_spec="Price data",
            success_rubric="Pass if price retrieved",
            order=1,
        )
        d = m.to_dict()
        assert d["description"] == "Retrieve stock price"
        assert d["evidence_spec"] == "Price data"
        assert d["success_rubric"] == "Pass if price retrieved"
        assert d["order"] == 1
        assert isinstance(d["id"], str)
        uuid.UUID(d["id"])  # should not raise

    def test_milestone_from_dict(self) -> None:
        raw = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "description": "Compute ROI",
            "evidence_spec": "ROI value",
            "success_rubric": "Pass if ROI > 10%",
            "order": 2,
        }
        m = Milestone.from_dict(raw)
        assert m.id == uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        assert m.description == "Compute ROI"
        assert m.evidence_spec == "ROI value"
        assert m.success_rubric == "Pass if ROI > 10%"
        assert m.order == 2

    def test_milestone_dict_roundtrip(self) -> None:
        original = Milestone(
            description="Verify compliance",
            evidence_spec="Compliance report",
            success_rubric="All checks pass",
            order=0,
        )
        d = original.to_dict()
        restored = Milestone.from_dict(d)
        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.evidence_spec == original.evidence_spec
        assert restored.success_rubric == original.success_rubric
        assert restored.order == original.order

    def test_milestone_repr(self) -> None:
        m = Milestone(
            description="Generate report",
            evidence_spec="PDF output",
            success_rubric="PDF exists",
            order=3,
        )
        r = repr(m)
        assert "Milestone" in r
        assert "order=3" in r
        assert "Generate report" in r


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSplitIntoClauses:
    def test_split_on_and(self) -> None:
        clauses = _split_into_clauses("Analyze revenue and compare to last year")
        assert len(clauses) == 2
        assert "analyze revenue" in clauses[0].lower()
        assert "compare to last year" in clauses[1].lower()

    def test_split_on_but(self) -> None:
        clauses = _split_into_clauses(
            "Fetch the data but do not process it yet"
        )
        assert len(clauses) == 2

    def test_split_on_then(self) -> None:
        clauses = _split_into_clauses(
            "Retrieve the file then analyze its contents"
        )
        assert len(clauses) == 2

    def test_split_on_sentence_boundary(self) -> None:
        clauses = _split_into_clauses(
            "First, analyze the revenue. Second, compare it to last year."
        )
        assert len(clauses) == 2

    def test_split_finally(self) -> None:
        clauses = _split_into_clauses(
            "Retrieve data, analyze it, and finally generate a report"
        )
        # Should produce at least 2 clauses (compound first + final clause)
        assert len(clauses) >= 2
        last_clause = clauses[-1].lower()
        assert "report" in last_clause or "generate" in last_clause

    def test_single_clause_no_split(self) -> None:
        clauses = _split_into_clauses("Analyze quarterly revenue figures")
        assert len(clauses) == 1
        assert "quarterly revenue" in clauses[0].lower()

    def test_empty_string(self) -> None:
        clauses = _split_into_clauses("")
        assert clauses == []

    def test_whitespace_normalised(self) -> None:
        clauses = _split_into_clauses(
            "  Analyze   revenue  and   compare  it  "
        )
        assert len(clauses) == 2
        for c in clauses:
            assert "  " not in c


class TestIdentifyVerb:
    def test_analyze(self) -> None:
        assert _identify_verb("Analyze the revenue figures") == "analyze"

    def test_compare(self) -> None:
        assert _identify_verb("Compare Q3 to Q3 last year") == "compare"

    def test_retrieve(self) -> None:
        assert _identify_verb("Retrieve the stock price") == "retrieve"

    def test_generate(self) -> None:
        assert _identify_verb("Generate a PDF report") == "generate"

    def test_no_match(self) -> None:
        assert _identify_verb("Do something vague") is None

    def test_case_insensitive(self) -> None:
        assert _identify_verb("ANALYZE revenue") == "analyze"
        assert _identify_verb("AnAlYze the data") == "analyze"


class TestMakeEvidenceSpec:
    def test_analyze_evidence(self) -> None:
        spec = _make_evidence_spec("Analyze the revenue")
        assert "analytical" in spec.lower() or "analysis" in spec.lower()

    def test_compute_evidence(self) -> None:
        spec = _make_evidence_spec("Compute the total cost")
        assert "calculation" in spec.lower() or "numerical" in spec.lower()

    def test_unknown_verb(self) -> None:
        spec = _make_evidence_spec("Do something vague")
        assert len(spec) > 0


class TestMakeSuccessRubric:
    def test_rubric_mentions_order(self) -> None:
        rubric = _make_success_rubric("Analyze revenue", order=0)
        assert "Milestone 1" in rubric
        assert "PASS" in rubric
        assert "FAIL" in rubric
        assert "UNCERTAIN" in rubric


# ---------------------------------------------------------------------------
# MilestoneDecomposer.decompose() tests
# ---------------------------------------------------------------------------


class TestDecomposeSimple:
    def test_single_action(self, decomposer: MilestoneDecomposer) -> None:
        milestones = decomposer.decompose("Analyze Q3 revenue")
        assert len(milestones) == 1
        assert milestones[0].order == 0
        assert "analyze" in milestones[0].description.lower()
        assert milestones[0].evidence_spec != ""
        assert milestones[0].success_rubric != ""

    def test_two_actions(self, decomposer: MilestoneDecomposer) -> None:
        milestones = decomposer.decompose(
            "Analyze Q3 revenue and compare to Q3 last year"
        )
        assert len(milestones) == 2
        assert milestones[0].order == 0
        assert milestones[1].order == 1
        # Order must be sequential
        orders = [m.order for m in milestones]
        assert orders == sorted(orders)

    def test_compare_task(self, decomposer: MilestoneDecomposer) -> None:
        milestones = decomposer.decompose(
            "Analyze Q3 revenue and compare to Q3 last year"
        )
        assert len(milestones) == 2
        desc_lower = " ".join(m.description.lower() for m in milestones)
        assert "analyze" in desc_lower or "q3 revenue" in desc_lower
        assert "compare" in desc_lower or "last year" in desc_lower

    def test_empty_task_raises(self, decomposer: MilestoneDecomposer) -> None:
        with pytest.raises(ValueError, match="empty"):
            decomposer.decompose("")
        with pytest.raises(ValueError, match="empty"):
            decomposer.decompose("   ")

    def test_ids_are_unique(self, decomposer: MilestoneDecomposer) -> None:
        milestones = decomposer.decompose(
            "Analyze revenue and compute costs and generate a report"
        )
        ids = [m.id for m in milestones]
        assert len(ids) == len(set(ids))


class TestDecomposeComplex:
    def test_multi_step_workflow(self, decomposer: MilestoneDecomposer) -> None:
        task = (
            "Retrieve the quarterly sales data, then analyze trends, "
            "compare to the same quarter last year, and finally "
            "generate a summary report with recommendations."
        )
        milestones = decomposer.decompose(task)
        assert len(milestones) >= 3
        orders = [m.order for m in milestones]
        assert orders == list(range(len(orders)))

    def test_all_milestones_have_required_fields(
        self, decomposer: MilestoneDecomposer
    ) -> None:
        task = "Fetch data, analyze it, verify results, and generate output."
        milestones = decomposer.decompose(task)
        for m in milestones:
            assert isinstance(m.id, uuid.UUID)
            assert isinstance(m.description, str)
            assert len(m.description) > 0
            assert isinstance(m.evidence_spec, str)
            assert len(m.evidence_spec) > 0
            assert isinstance(m.success_rubric, str)
            assert len(m.success_rubric) > 0
            assert isinstance(m.order, int)

    def test_sequential_ordering(self, decomposer: MilestoneDecomposer) -> None:
        task = "First retrieve data. Then analyze it. Then generate a report."
        milestones = decomposer.decompose(task)
        orders = [m.order for m in milestones]
        assert orders == sorted(orders)
        # Strictly increasing by 1
        assert orders == list(range(len(milestones)))


# ---------------------------------------------------------------------------
# MilestoneDecomposer.refine() tests
# ---------------------------------------------------------------------------


class TestRefine:
    def test_refine_returns_same_when_no_key(self) -> None:
        """refine() should return original milestones when no API key is set."""
        decomposer = MilestoneDecomposer(openai_api_key=None)
        original = [
            Milestone(
                description="Analyze revenue",
                evidence_spec="Revenue figures",
                success_rubric="Pass if analyzed",
                order=0,
            )
        ]
        result = decomposer.refine(original, "Analyze revenue")
        # Must not crash; returns original list (copied)
        assert len(result) == 1
        assert result[0].description == original[0].description

    def test_refine_returns_same_with_dummy_key(self) -> None:
        """refine() with a fake key should gracefully fall back to originals."""
        decomposer = MilestoneDecomposer(openai_api_key="sk-fake-key")
        original = [
            Milestone(
                description="Retrieve stock price",
                evidence_spec="Price data",
                success_rubric="Pass if retrieved",
                order=0,
            )
        ]
        # Will attempt API call (may fail), should fall back to originals
        result = decomposer.refine(original, "Retrieve stock price")
        assert len(result) == 1

    def test_refine_empty_list(self, decomposer: MilestoneDecomposer) -> None:
        """refine() on empty list should return empty list."""
        result = decomposer.refine([], "Any task")
        assert result == []

    def test_refine_preserves_order_field(self) -> None:
        """After refine (real or fallback), order field must remain valid."""
        decomposer = MilestoneDecomposer(openai_api_key=None)
        milestones = [
            Milestone(
                description="Step one",
                evidence_spec="Evidence one",
                success_rubric="Pass one",
                order=0,
            ),
            Milestone(
                description="Step two",
                evidence_spec="Evidence two",
                success_rubric="Pass two",
                order=1,
            ),
        ]
        result = decomposer.refine(milestones, "Step one then step two")
        orders = [m.order for m in result]
        assert orders == [0, 1]


# ---------------------------------------------------------------------------
# Serialisation roundtrip through MilestoneDecomposer.decompose
# ---------------------------------------------------------------------------


class TestDecomposeRoundtrip:
    def test_decompose_and_serialize(self, decomposer: MilestoneDecomposer) -> None:
        task = "Analyze revenue and compare to last year."
        milestones = decomposer.decompose(task)
        # Serialize each milestone
        for m in milestones:
            d = m.to_dict()
            restored = Milestone.from_dict(d)
            assert restored.id == m.id
            assert restored.description == m.description
            assert restored.evidence_spec == m.evidence_spec
            assert restored.success_rubric == m.success_rubric
            assert restored.order == m.order

    def test_milestone_order_survives_serialization(
        self, decomposer: MilestoneDecomposer
    ) -> None:
        task = "Fetch data, verify it, and generate a report."
        milestones = decomposer.decompose(task)
        for m in milestones:
            restored = Milestone.from_dict(m.to_dict())
            assert restored.order == m.order
