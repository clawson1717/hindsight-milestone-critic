"""
Tests for the HiMCA CLI.
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli import (
    _BENCHMARK_TASKS,
    _build_mock_evidence_provider,
    _build_parser,
    _cmd_benchmark,
    _cmd_critic,
    _cmd_hindsight,
    _cmd_run,
    _format_verdict,
    main,
)
from src.evidence import Evidence
from src.loop import HiMCALoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_milestones():
    """Create sample milestones for testing."""
    from src.decompose import Milestone

    return [
        Milestone(
            description="Analyze revenue data",
            evidence_spec="Analytical notes",
            success_rubric="Revenue analysis completed",
            order=0,
            id=uuid.uuid4(),
        ),
        Milestone(
            description="Compare year-over-year growth",
            evidence_spec="Comparison table",
            success_rubric="Growth comparison done",
            order=1,
            id=uuid.uuid4(),
        ),
    ]


@pytest.fixture
def sample_loop_result():
    """Create a sample LoopResult."""
    from src.critic import MilestoneCriticResult, MilestoneVerdict
    from src.decompose import Milestone
    from src.evidence import Evidence
    from src.hindsight import HindsightEntry
    from src.loop import LoopResult

    m = Milestone(
        description="Analyze revenue data",
        evidence_spec="Analytical notes",
        success_rubric="Revenue analysis completed",
        order=0,
    )
    result = MilestoneCriticResult(
        milestone_id=str(m.id),
        verdict=MilestoneVerdict.PASS,
        reasoning="PASS: Strong evidence.",
        confidence=0.85,
    )
    evidence = Evidence(
        milestone_id=str(m.id),
        text_snippets=["Revenue analysis completed successfully."],
        citations=[],
        intermediate_results=[],
        confidence=0.9,
    )
    entry = HindsightEntry(
        original_task="Analyze revenue",
        failed_milestone=m,
        failed_milestone_index=0,
        collected_evidence=evidence,
        critic_result=result,
        trajectory=[],
        hindsight_label="What if [Analyze revenue data] was the actual goal?",
    )
    return LoopResult(
        success=True,
        attempts=1,
        milestones_verified=[m],
        milestone_verdicts=[[result]],
        hindsight_entries=[entry],
        final_verdict="All 1 milestone(s) PASSED on attempt 1. Task completed successfully.",
    )


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------


class TestArgParser:
    def test_parser_run_command(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--task", "Analyze revenue"])
        assert args.command == "run"
        assert args.task == "Analyze revenue"
        assert args.max_attempts == 3
        assert args.verbose is False

    def test_parser_run_with_max_attempts(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--task", "Analyze revenue", "--max-attempts", "5"])
        assert args.max_attempts == 5

    def test_parser_run_with_max_attempts_short(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--task", "Analyze revenue", "-m", "4"])
        assert args.max_attempts == 4

    def test_parser_run_with_verbose(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--task", "Analyze revenue", "--verbose"])
        assert args.verbose is True

    def test_parser_run_verbose_short_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--task", "Analyze revenue", "-v"])
        assert args.verbose is True

    def test_parser_run_task_short_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "-t", "Analyze revenue"])
        assert args.task == "Analyze revenue"

    def test_parser_critic_command(self):
        parser = _build_parser()
        mid = str(uuid.uuid4())
        args = parser.parse_args(["critic", "--task", "Analyze revenue", "--milestone-id", mid])
        assert args.command == "critic"
        assert args.task == "Analyze revenue"
        assert args.milestone_id == mid

    def test_parser_critic_with_index(self):
        parser = _build_parser()
        args = parser.parse_args(["critic", "--task", "Analyze revenue", "--milestone-id", "0"])
        assert args.milestone_id == "0"

    def test_parser_critic_task_short_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["critic", "-t", "Analyze revenue"])
        assert args.task == "Analyze revenue"

    def test_parser_hindsight_query(self):
        parser = _build_parser()
        args = parser.parse_args(["hindsight", "--query", "revenue analysis"])
        assert args.command == "hindsight"
        assert args.query == "revenue analysis"
        assert args.list is False
        assert args.top_k == 5

    def test_parser_hindsight_query_short(self):
        parser = _build_parser()
        args = parser.parse_args(["hindsight", "-q", "revenue analysis"])
        assert args.query == "revenue analysis"

    def test_parser_hindsight_list(self):
        parser = _build_parser()
        args = parser.parse_args(["hindsight", "--list"])
        assert args.command == "hindsight"
        assert args.list is True
        assert args.query is None

    def test_parser_hindsight_list_short(self):
        parser = _build_parser()
        args = parser.parse_args(["hindsight", "-l"])
        assert args.list is True

    def test_parser_hindsight_top_k(self):
        parser = _build_parser()
        args = parser.parse_args(["hindsight", "-q", "test", "-k", "10"])
        assert args.top_k == 10

    def test_parser_benchmark_command(self):
        parser = _build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"

    def test_parser_benchmark_no_header(self):
        parser = _build_parser()
        args = parser.parse_args(["benchmark", "--no-header"])
        assert args.no_header is True

    def test_parser_missing_task_raises(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])

    def test_parser_missing_milestone_id_not_required(self):
        parser = _build_parser()
        args = parser.parse_args(["critic", "--task", "Analyze revenue"])
        assert args.milestone_id is None

    def test_parser_unknown_command(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown"])


# ---------------------------------------------------------------------------
# CLI main tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_unknown_command_exits_2(self):
        exit_code = main(["unknown"])
        assert exit_code == 2

    def test_main_no_args_shows_help(self):
        exit_code = main([])
        assert exit_code == 0  # Shows quick-start help


# ---------------------------------------------------------------------------
# Mock evidence provider tests
# ---------------------------------------------------------------------------


class TestMockEvidenceProvider:
    def test_build_mock_evidence_returns_evidence_object(self, sample_milestones):
        provider = _build_mock_evidence_provider("Analyze revenue")
        ev_list = provider(sample_milestones, attempt=1, hindsight_entries=[])
        assert isinstance(ev_list, list)
        assert all(isinstance(ev, Evidence) for ev in ev_list)

    def test_build_mock_evidence_has_snippets(self, sample_milestones):
        provider = _build_mock_evidence_provider("Analyze revenue")
        ev_list = provider(sample_milestones, attempt=1, hindsight_entries=[])
        assert len(ev_list) > 0
        assert len(ev_list[0].text_snippets) > 0

    def test_build_mock_evidence_uses_task_text(self):
        provider = _build_mock_evidence_provider("My special revenue task")
        # Call with dummy milestones whose descriptions are reflected in evidence
        from src.decompose import Milestone
        dummy_milestones = [
            Milestone(
                description="Special revenue milestone",
                evidence_spec="Test spec",
                success_rubric="Test rubric",
                order=0,
            )
        ]
        ev_list = provider(dummy_milestones, attempt=1, hindsight_entries=[])
        assert any(
            "special" in s.lower() or "revenue" in s.lower()
            for ev in ev_list
            for s in ev.text_snippets
        )


# ---------------------------------------------------------------------------
# Run command tests
# ---------------------------------------------------------------------------


class TestCmdRun:
    def test_run_command_success_path(self, sample_loop_result):
        args = _build_parser().parse_args(["run", "--task", "Analyze revenue"])
        with patch.object(HiMCALoop, "run", return_value=sample_loop_result):
            exit_code = _cmd_run(args)
            assert exit_code == 0

    def test_run_command_failure_path(self, sample_loop_result):
        sample_loop_result.success = False
        args = _build_parser().parse_args(["run", "--task", "Analyze revenue"])
        with patch.object(HiMCALoop, "run", return_value=sample_loop_result):
            exit_code = _cmd_run(args)
            assert exit_code == 1


# ---------------------------------------------------------------------------
# Benchmark command tests
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_has_builtin_tasks(self):
        assert len(_BENCHMARK_TASKS) >= 5

    def test_benchmark_tasks_all_have_task(self):
        for t in _BENCHMARK_TASKS:
            assert "task" in t
            assert isinstance(t["task"], str)
            assert len(t["task"]) > 0

    def test_benchmark_tasks_all_have_expected_milestones(self):
        for t in _BENCHMARK_TASKS:
            assert "expected_milestones" in t
            assert isinstance(t["expected_milestones"], int)

    def test_benchmark_command_exists(self):
        args = _build_parser().parse_args(["benchmark"])
        assert args.command == "benchmark"


# ---------------------------------------------------------------------------
# Hindsight command tests
# ---------------------------------------------------------------------------


class TestHindsight:
    def test_hindsight_list_empty_store(self):
        args = _build_parser().parse_args(["hindsight", "--list"])
        # Should not raise, returns 0
        exit_code = _cmd_hindsight(args)
        assert exit_code == 0

    def test_hindsight_query_empty_store(self):
        args = _build_parser().parse_args(["hindsight", "--query", "revenue analysis"])
        exit_code = _cmd_hindsight(args)
        assert exit_code == 0

    def test_hindsight_no_args_returns_2(self):
        args = _build_parser().parse_args(["hindsight"])
        exit_code = _cmd_hindsight(args)
        assert exit_code == 2


# ---------------------------------------------------------------------------
# Verdict formatting tests
# ---------------------------------------------------------------------------


class TestVerdictFormatting:
    def test_format_verdict_pass(self):
        from src.critic import MilestoneVerdict

        result = _format_verdict(MilestoneVerdict.PASS)
        assert "PASS" in result

    def test_format_verdict_fail(self):
        from src.critic import MilestoneVerdict

        result = _format_verdict(MilestoneVerdict.FAIL)
        assert "FAIL" in result

    def test_format_verdict_uncertain(self):
        from src.critic import MilestoneVerdict

        result = _format_verdict(MilestoneVerdict.UNCERTAIN)
        assert "UNCERTAIN" in result
