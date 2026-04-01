"""
HiMCA CLI — Command-line interface for the Hindsight Milestone Critic Agent.

Provides four commands:
    run       — Run a task through the full HiMCA loop
    critic    — Evaluate a task/milestone with the critic directly
    hindsight — Query and display hindsight entries from the store
    benchmark — Run a built-in benchmark of milestone tasks

Usage:
    python -m src.cli run --task "Analyze revenue and generate a report"
    python -m src.cli critic --task "..." --milestone-id "..."
    python -m src.cli hindsight --list
    python -m src.cli hindsight --query "revenue"
    python -m src.cli benchmark
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from src.critic import MilestoneCritic, MilestoneCriticResult, MilestoneVerdict
from src.decompose import Milestone, MilestoneDecomposer
from src.evidence import Evidence, EvidenceCollector
from src.hindsight import HindsightEntry, HindsightStore
from src.loop import HiMCALoop, LoopResult

# ---------------------------------------------------------------------------
# Console singleton
# ---------------------------------------------------------------------------

console = Console()


# ---------------------------------------------------------------------------
# Verdict emoji / colour helpers
# ---------------------------------------------------------------------------

_VERDICT_EMOJI = {
    MilestoneVerdict.PASS: "✅",
    MilestoneVerdict.FAIL: "❌",
    MilestoneVerdict.UNCERTAIN: "⏳",
}

_VERDICT_COLOR = {
    MilestoneVerdict.PASS: "green",
    MilestoneVerdict.FAIL: "red",
    MilestoneVerdict.UNCERTAIN: "yellow",
}


def _format_verdict(verdict: MilestoneVerdict) -> str:
    return f"{_VERDICT_EMOJI.get(verdict, '❓')} {verdict.value.upper()}"


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

def _has_openai_key() -> bool:
    """Check if OPENAI_API_KEY is set in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Mock evidence provider (used when no API key)
# ---------------------------------------------------------------------------

def _build_mock_evidence_provider(task: str):
    """
    Build a mock evidence provider that returns simulated evidence.

    Used when OPENAI_API_KEY is not set, so the CLI remains functional
    for demos and testing.
    """

    def provider(
        milestones: List[Milestone],
        attempt: int,
        hindsight_entries: List[HindsightEntry],
    ) -> List[Evidence]:
        results: List[Evidence] = []
        for m in milestones:
            # Simulate some evidence: use milestone description as a text snippet
            confidence = 0.5 + 0.3 * (attempt - 1)  # improve slightly each attempt
            confidence = min(1.0, confidence)
            results.append(
                Evidence(
                    milestone_id=str(m.id),
                    text_snippets=[
                        f"[Attempt {attempt}] Evidence for milestone: {m.description[:60]}"
                    ],
                    citations=[],
                    intermediate_results=[],
                    confidence=confidence,
                )
            )
        return results

    return provider


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_milestone_table(
    milestones: List[Milestone],
    results: List[MilestoneCriticResult],
    title: str = "Milestone Results",
) -> None:
    """Print a formatted table of milestones and their verdicts."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Milestone", width=40)
    table.add_column("Verdict", width=12)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Reasoning (truncated)", width=40)

    for i, (m, r) in enumerate(zip(milestones, results)):
        emoji = _VERDICT_EMOJI.get(r.verdict, "❓")
        reasoning_snippet = (
            r.reasoning[:60] + "…" if len(r.reasoning) > 60 else r.reasoning
        )
        table.add_row(
            str(i + 1),
            m.description[:40],
            f"{emoji} {r.verdict.value}",
            f"{r.confidence:.2f}",
            reasoning_snippet,
        )

    console.print(table)


def _load_evidence(arg: Optional[str]) -> Optional[Evidence]:
    """Load evidence from a file path or inline JSON string."""
    if not arg:
        return None
    path = Path(arg)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Evidence.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            console.print(
                f"[yellow]⚠️  Could not parse evidence file; ignoring.[/yellow]"
            )
            return None
    # Try as inline JSON
    try:
        data = json.loads(arg)
        return Evidence.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        console.print(
            f"[yellow]⚠️  Could not parse --evidence as inline JSON; ignoring.[/yellow]"
        )
        return None


# ---------------------------------------------------------------------------
# Built-in benchmark tasks
# ---------------------------------------------------------------------------

_BENCHMARK_TASKS: List[Dict[str, Any]] = [
    {
        "task": "Analyze Q3 revenue and generate a summary report.",
        "expected_milestones": 2,
        "description": "Analyze Q3 revenue and generate a summary report.",
    },
    {
        "task": "Retrieve stock prices for Apple and Google, then compare them.",
        "expected_milestones": 2,
        "description": "Retrieve and compare stock prices.",
    },
    {
        "task": "Fetch user feedback data, categorize it, and produce a PDF report.",
        "expected_milestones": 3,
        "description": "Fetch, categorize, and report user feedback.",
    },
    {
        "task": "Calculate the average response time from API logs and verify it meets SLA.",
        "expected_milestones": 2,
        "description": "Calculate and verify API response time SLA.",
    },
    {
        "task": "Identify all high-severity bugs in the repository, then prioritize them by impact.",
        "expected_milestones": 2,
        "description": "Identify and prioritize high-severity bugs.",
    },
    {
        "task": "Collect metrics from three data sources, aggregate them, and plot a chart.",
        "expected_milestones": 3,
        "description": "Collect, aggregate, and visualize metrics.",
    },
    {
        "task": "Query the database for inactive users, send them re-engagement emails, and log the results.",
        "expected_milestones": 3,
        "description": "Query inactive users, email, and log.",
    },
    {
        "task": "Evaluate two ML models on the test set and recommend the better one.",
        "expected_milestones": 2,
        "description": "Evaluate and recommend ML model.",
    },
    {
        "task": "Extract key terms from 10 research papers and synthesize them into a literature review.",
        "expected_milestones": 2,
        "description": "Extract and synthesize research paper terms.",
    },
    {
        "task": "Verify system configuration, ensure all services are healthy, and generate a health report.",
        "expected_milestones": 3,
        "description": "Verify, check health, and report.",
    },
]


def _run_benchmark_task(
    task_item: Dict[str, Any],
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Run a single benchmark task and return metrics."""
    task = task_item["task"]
    evidence_provider = _build_mock_evidence_provider(task)

    loop = HiMCALoop(
        decomposer=MilestoneDecomposer(),
        critic=MilestoneCritic(),
        hindsight_store=HindsightStore(),
        evidence_provider=evidence_provider,
        max_attempts=max_attempts,
    )

    result = loop.run(task, max_attempts=max_attempts)

    last_verdicts = result.milestone_verdicts[-1] if result.milestone_verdicts else []
    pass_count = sum(1 for v in last_verdicts if v.verdict == MilestoneVerdict.PASS)
    total_milestones = len(last_verdicts)
    milestone_pass_rate = pass_count / total_milestones if total_milestones > 0 else 0.0

    return {
        "task": task,
        "description": task_item["description"],
        "success": result.success,
        "attempts": result.attempts,
        "milestone_count": total_milestones,
        "milestone_pass_count": pass_count,
        "milestone_pass_rate": milestone_pass_rate,
        "verdicts": [v.verdict.value for v in last_verdicts],
    }


# ---------------------------------------------------------------------------
# Command: run
# ---------------------------------------------------------------------------

def _build_run_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "run",
        help="Run a task through the full HiMCA loop.",
        description="Decompose a task into milestones, run the loop, and display results.",
    )
    parser.add_argument("--task", "-t", required=True, help="Natural-language task description.")
    parser.add_argument(
        "--max-attempts", "-m", type=int, default=3,
        help="Maximum loop iterations (default: 3).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output.")
    parser.add_argument("--no-header", action="store_true", help="Suppress the welcome banner.")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    task = args.task.strip()
    max_attempts = args.max_attempts
    verbose = args.verbose

    if not args.no_header:
        console.print(Panel.fit(
            "[bold cyan]HiMCA CLI — Run Mode[/bold cyan]\n"
            f"Task: {task}\nMax attempts: {max_attempts}",
            border_style="cyan",
        ))

    if max_attempts < 1:
        console.print("[red]✗ --max-attempts must be at least 1.[/red]")
        return 2

    if not _has_openai_key():
        console.print(
            "[yellow]⚠️  OPENAI_API_KEY not set. Using mock evidence provider.[/yellow]"
        )
        console.print("[dim]Set OPENAI_API_KEY for real LLM-based evaluation.[/dim]\n")

    decomposer = MilestoneDecomposer()
    critic = MilestoneCritic()
    store = HindsightStore()
    evidence_provider = _build_mock_evidence_provider(task)

    loop = HiMCALoop(
        decomposer=decomposer,
        critic=critic,
        hindsight_store=store,
        evidence_provider=evidence_provider,
        max_attempts=max_attempts,
    )

    console.print("[cyan]Decomposing task…[/cyan]")
    try:
        milestones = decomposer.decompose(task)
    except Exception as e:
        console.print(f"[red]✗ Failed to decompose task:[/red] {e}")
        return 1
    if not milestones:
        console.print("[yellow]⚠️  No milestones could be decomposed from the task.[/yellow]")
        return 1
    console.print(f"[green]✓[/green] {len(milestones)} milestone(s) identified.\n")

    if verbose:
        for i, m in enumerate(milestones):
            console.print(f"  [{i+1}] {m.description}")
        console.print()

    console.print("[cyan]Running HiMCA loop…[/cyan]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Running", total=max_attempts)
            result = loop.run(task, max_attempts=max_attempts)
            progress.update(task_id, completed=max_attempts, description="Done.")
    except Exception as e:
        console.print(f"[red]✗ HiMCA loop failed:[/red] {e}")
        return 1

    console.print()

    for attempt_idx, verdicts in enumerate(result.milestone_verdicts, 1):
        attempt_label = f"Attempt {attempt_idx}"
        console.print(f"[bold]{attempt_label}[/bold]")
        _print_milestone_table(
            result.milestones_verified,
            verdicts,
            title=f"{attempt_label} Milestone Verdicts",
        )
        console.print()

    if result.success:
        console.print(Panel.fit(
            f"[bold green]✓ SUCCESS[/bold green]\n{result.final_verdict}\n"
            f"Attempts: {result.attempts} / {max_attempts}",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]✗ FAILED[/bold red]\n{result.final_verdict}\n"
            f"Attempts: {result.attempts} / {max_attempts}",
            border_style="red",
        ))

    if result.hindsight_entries:
        console.print(
            f"\n[dim]{len(result.hindsight_entries)} hindsight entry/entries stored in session store.[/dim]"
        )
        try:
            store.export()
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not persist hindsight store: {e}[/yellow]")

    return 0 if result.success else 1


# ---------------------------------------------------------------------------
# Command: critic
# ---------------------------------------------------------------------------

def _build_critic_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "critic",
        help="Evaluate a single milestone with the critic directly.",
        description="Decompose a task, pick a milestone, and run it through MilestoneCritic.",
    )
    parser.add_argument("--task", "-t", required=True, help="Natural-language task description.")
    parser.add_argument(
        "--milestone-id", "-i",
        help="UUID of the milestone to evaluate (partial ID works).",
    )
    parser.add_argument(
        "--milestone-index", "-n", type=int, default=None,
        help="1-based index of the milestone to evaluate (alternative to --milestone-id).",
    )
    parser.add_argument(
        "--evidence", "-e", default=None,
        help="JSON evidence file path or inline JSON string.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full reasoning output.")
    return parser


def _cmd_critic(args: argparse.Namespace) -> int:
    task = args.task.strip()
    milestone_id_arg = args.milestone_id
    milestone_index = args.milestone_index
    evidence_arg = args.evidence
    verbose = args.verbose

    console.print(Panel.fit(
        "[bold cyan]HiMCA CLI — Critic Mode[/bold cyan]\nTask: " + task,
        border_style="cyan",
    ))

    decomposer = MilestoneDecomposer()
    milestones = decomposer.decompose(task)

    if not milestones:
        console.print("[red]No milestones could be decomposed from the task.[/red]")
        return 2

    target_milestone: Optional[Milestone] = None

    if milestone_id_arg:
        for m in milestones:
            if str(m.id).startswith(milestone_id_arg):
                target_milestone = m
                break
        if not target_milestone:
            console.print(
                f"[red]No milestone found matching ID prefix '{milestone_id_arg}'.[/red]"
            )
            console.print("[dim]Available IDs:[/dim]")
            for m in milestones:
                console.print(f"  {m.id}  {m.description[:60]}")
            return 1

    elif milestone_index is not None:
        idx = milestone_index - 1
        if 0 <= idx < len(milestones):
            target_milestone = milestones[idx]
        else:
            console.print(
                f"[red]Milestone index {milestone_index} out of range (1–{len(milestones)}).[/red]"
            )
            return 2
    else:
        target_milestone = milestones[0]
        console.print(
            "[dim]No --milestone-id or --milestone-index given; using milestone 1.[/dim]"
        )

    loaded = _load_evidence(evidence_arg)
    if loaded:
        evidence = loaded
        console.print("[green]✓[/green] Evidence loaded from argument.")
    else:
        evidence = Evidence(
            milestone_id=str(target_milestone.id),
            text_snippets=[
                f"Evidence for milestone: {target_milestone.description[:80]}"
            ],
            citations=[],
            intermediate_results=[],
            confidence=0.75,
        )
        console.print("[yellow]Using simulated evidence (no --evidence provided).[/yellow]")

    console.print(f"\n[bold]Evaluating Milestone:[/bold]")
    console.print(f"  ID:          {target_milestone.id}")
    console.print(f"  Description: {target_milestone.description}")
    console.print(f"  Rubric:      {target_milestone.success_rubric[:80]}…\n")

    critic = MilestoneCritic()
    result = critic.critique([target_milestone], [evidence])[0]

    color = _VERDICT_COLOR.get(result.verdict, "white")
    emoji = _VERDICT_EMOJI.get(result.verdict, "❓")
    verdict_text = (
        f"{emoji} {result.verdict.value.upper()} "
        f"(confidence={result.confidence:.2f})"
    )

    console.print(Panel.fit(
        f"[bold {color}]{verdict_text}[/bold {color}]",
        title="Verdict",
        border_style=color,
    ))

    if verbose:
        console.print(f"\n[bold]Full Reasoning:[/bold]")
        console.print(textwrap.indent(result.reasoning, "  "))

    scores = Table(title="Evidence Sub-Scores", show_header=False)
    scores.add_column("Dimension", width=20)
    scores.add_column("Score", justify="right", width=8)
    scores.add_row("Quality", f"{result.evidence_quality:.3f}")
    scores.add_row("Sufficiency", f"{result.evidence_sufficiency:.3f}")
    scores.add_row("Consistency", f"{result.evidence_consistency:.3f}")
    console.print(scores)

    if result.violations:
        console.print(
            f"\n[bold red]Boundary Violations ({len(result.violations)}):[/bold red]"
        )
        for v in result.violations:
            console.print(
                f"  [{v.severity.upper():>6}] {v.violation_type}: {v.description}"
            )
    else:
        console.print("\n[dim]No boundary violations detected.[/dim]")

    return 0 if result.verdict == MilestoneVerdict.PASS else 1


# ---------------------------------------------------------------------------
# Command: hindsight
# ---------------------------------------------------------------------------

def _build_hindsight_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "hindsight",
        help="Query and display hindsight entries from the store.",
        description="Retrieve hindsight entries by query or list all entries.",
    )
    parser.add_argument("--query", "-q", default=None, help="Text query to search hindsight entries.")
    parser.add_argument("--list", "-l", action="store_true", help="List all hindsight entries.")
    parser.add_argument("--show", "-s", action="store_true", help="Show detailed entry (use with --list).")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Maximum entries to return (default: 5).")
    parser.add_argument("--store-dir", "-d", default=None, help="Path to the hindsight store directory.")
    parser.add_argument("--stats", action="store_true", help="Show store statistics.")
    return parser


def _print_hindsight_entry(entry: HindsightEntry, detailed: bool = False) -> None:
    """Print a single hindsight entry."""
    panel_lines = [
        f"[bold]Task:[/bold] {entry.original_task[:80]}",
        f"[bold]Milestone:[/bold] {entry.failed_milestone.description[:80]}",
        f"[bold]Hindsight Label:[/bold] {entry.hindsight_label[:80]}",
        f"[bold]Verdict:[/bold] {_format_verdict(entry.critic_result.verdict)} "
        f"(confidence={entry.critic_result.confidence:.2f})",
        f"[bold]Created:[/bold] {entry.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"[bold]Episode ID:[/bold] {entry.episode_id}",
    ]
    if detailed:
        panel_lines.extend([
            f"[bold]Entry ID:[/bold] {entry.entry_id}",
            f"[bold]Milestone Index:[/bold] {entry.failed_milestone_index}",
            f"[bold]Keywords:[/bold] {', '.join(entry.retrieval_keywords[:15])}",
            f"[bold]Evidence snippets:[/bold]",
        ])
        for snippet in entry.collected_evidence.text_snippets[:5]:
            panel_lines.append(f"  • {snippet[:80]}")
        if entry.critic_result.reasoning:
            panel_lines.append("[bold]Reasoning:[/bold]")
            panel_lines.append(f"  {entry.critic_result.reasoning[:200]}")

    console.print(Panel("\n".join(panel_lines), border_style="blue"))
    console.print()


def _cmd_hindsight(args: argparse.Namespace) -> int:
    store_dir = Path(args.store_dir) if args.store_dir else None
    store = HindsightStore(store_dir=store_dir)

    loaded_count = store.load()
    if loaded_count > 0:
        console.print(f"[dim]Loaded {loaded_count} persisted entry/entries.[/dim]\n")

    if not args.list and not args.query and not args.stats:
        # No actionable hindsight command given — show usage error
        console.print(
            "[yellow]⚠️  Use --list, --query, or --stats. "
            "Run 'himca hindsight --help' for usage.[/yellow]"
        )
        return 2

    if args.stats:
        stats = store.stats()
        console.print(Panel.fit(
            "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in stats.items()),
            title="Store Statistics",
            border_style="green",
        ))
        return 0

    if args.list:
        if not store:
            console.print("[yellow]Store is empty.[/yellow]")
            return 0
        console.print(f"[bold]All Hindsight Entries ({len(store)} total):[/bold]\n")
        for entry in store:
            _print_hindsight_entry(entry, detailed=args.show)
        return 0

    query = args.query
    if not query:
        stats = store.stats()
        console.print(Panel.fit(
            "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in stats.items()),
            title="Store Summary",
            border_style="green",
        ))
        return 0

    console.print(f"[cyan]Searching hindsight store for:[/cyan] {query!r}\n")
    results = store.retrieve(query, top_k=args.top_k)

    if not results:
        console.print("[yellow]No matching hindsight entries found.[/yellow]")
        return 0

    console.print(f"[green]Found {len(results)} matching entry/entries:[/green]\n")
    for entry in results:
        _print_hindsight_entry(entry, detailed=True)

    return 0


# ---------------------------------------------------------------------------
# Command: benchmark
# ---------------------------------------------------------------------------

def _build_benchmark_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "benchmark",
        help="Run a built-in benchmark of milestone tasks.",
        description="Run 10 simple tasks through HiMCALoop and report aggregate metrics.",
    )
    parser.add_argument(
        "--max-attempts", "-m", type=int, default=3,
        help="Max attempts per task (default: 3).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-task results.")
    parser.add_argument("--no-header", action="store_true", help="Suppress the benchmark header.")
    return parser


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if not args.no_header:
        console.print(Panel.fit(
            "[bold cyan]HiMCA Benchmark Suite[/bold cyan]\n"
            f"Tasks: {len(_BENCHMARK_TASKS)}  |  "
            f"Max attempts per task: {args.max_attempts}",
            border_style="cyan",
        ))
        if not _has_openai_key():
            console.print(
                "[yellow]⚠️  OPENAI_API_KEY not set — using mock evidence provider.[/yellow]\n"
            )

    console.print("[cyan]Running benchmark…[/cyan]\n")

    results: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Benchmark progress", total=len(_BENCHMARK_TASKS))
        for task_item in _BENCHMARK_TASKS:
            result = _run_benchmark_task(task_item, max_attempts=args.max_attempts)
            results.append(result)
            progress.advance(task_id)

    console.print()

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    total_attempts = sum(r["attempts"] for r in results)
    avg_attempts = total_attempts / total if total > 0 else 0.0
    total_milestones = sum(r["milestone_count"] for r in results)
    total_passed = sum(r["milestone_pass_count"] for r in results)
    milestone_pass_rate = total_passed / total_milestones if total_milestones > 0 else 0.0
    success_rate = success_count / total if total > 0 else 0.0

    table = Table(
        title="Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Task (truncated)", width=45)
    table.add_column("Success", width=8, justify="center")
    table.add_column("Attempts", width=9, justify="right")
    table.add_column("M-✅/Total", width=10, justify="right")
    table.add_column("Pass Rate", width=9, justify="right")

    for i, r in enumerate(results, 1):
        task_trunc = r["description"][:43]
        success_str = "✅" if r["success"] else "❌"
        pass_rate_str = f"{r['milestone_pass_rate']:.0%}"
        table.add_row(
            str(i),
            task_trunc,
            success_str,
            str(r["attempts"]),
            f"{r['milestone_pass_count']}/{r['milestone_count']}",
            pass_rate_str,
        )

    console.print(table)

    summary_lines = [
        f"[bold cyan]Tasks run:[/bold cyan]  {total}",
        f"[bold cyan]Fully successful:[/bold cyan]  {success_count} ({success_rate:.1%})",
        f"[bold cyan]Avg attempts:[/bold cyan]  {avg_attempts:.2f}",
        f"[bold cyan]Milestone pass rate:[/bold cyan]  {milestone_pass_rate:.1%}",
    ]

    console.print(Panel("\n".join(summary_lines), title="Summary", border_style="green"))

    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="himca",
        description="HiMCA — Hindsight Milestone Critic Agent CLI",
    )
    parser.add_argument("--version", "-V", action="store_true", help="Show version.")

    subparsers = parser.add_subparsers(dest="command", title="Commands")

    _build_run_parser(subparsers)
    _build_critic_parser(subparsers)
    _build_hindsight_parser(subparsers)
    _build_benchmark_parser(subparsers)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.

    Returns exit code:
        0 — success
        1 — task/evaluation failure
        2 — CLI usage error
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as se:
        return se.code if se.code is not None else 0

    if args.version:
        try:
            version = importlib.metadata.version("himca")
        except importlib.metadata.PackageNotFoundError:
            version = "dev"
        console.print(f"himca {version}")
        return 0

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "critic":
        return _cmd_critic(args)
    elif args.command == "hindsight":
        return _cmd_hindsight(args)
    elif args.command == "benchmark":
        return _cmd_benchmark(args)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
