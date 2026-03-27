"""
Milestone Decomposer — Task → List[Milestone]

Heuristic decomposition first (keyword extraction, action decomposition),
then LLM refinement if OPENAI_API_KEY is set.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

# Optional OpenAI client — only needed for refine()
try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Action verb / keyword patterns for heuristic decomposition
# ---------------------------------------------------------------------------
ACTION_PATTERNS = [
    re.compile(r"\banalyze\b", re.IGNORECASE),
    re.compile(r"\bretrieve\b", re.IGNORECASE),
    re.compile(r"\bcompute\b", re.IGNORECASE),
    re.compile(r"\bverify\b", re.IGNORECASE),
    re.compile(r"\bsynthesize\b", re.IGNORECASE),
    re.compile(r"\bevaluate\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bgenerate\b", re.IGNORECASE),
    re.compile(r"\bselect\b", re.IGNORECASE),
    re.compile(r"\bdecide\b", re.IGNORECASE),
    re.compile(r"\bfetch\b", re.IGNORECASE),
    re.compile(r"\bobtain\b", re.IGNORECASE),
    re.compile(r"\bidentify\b", re.IGNORECASE),
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\bbuild\b", re.IGNORECASE),
    re.compile(r"\bcreate\b", re.IGNORECASE),
    re.compile(r"\bdevelop\b", re.IGNORECASE),
    re.compile(r"\bdesign\b", re.IGNORECASE),
    re.compile(r"\bplan\b", re.IGNORECASE),
    re.compile(r"\bprepare\b", re.IGNORECASE),
    re.compile(r"\breview\b", re.IGNORECASE),
    re.compile(r"\bassess\b", re.IGNORECASE),
    re.compile(r"\bcheck\b", re.IGNORECASE),
    re.compile(r"\bensure\b", re.IGNORECASE),
    re.compile(r"\bestablish\b", re.IGNORECASE),
    re.compile(r"\bdefine\b", re.IGNORECASE),
    re.compile(r"\bmeasure\b", re.IGNORECASE),
    re.compile(r"\bcalculate\b", re.IGNORECASE),
    re.compile(r"\bderive\b", re.IGNORECASE),
    re.compile(r"\bplot\b", re.IGNORECASE),
    re.compile(r"\bchart\b", re.IGNORECASE),
    re.compile(r"\breport\b", re.IGNORECASE),
    re.compile(r"\bsummarize\b", re.IGNORECASE),
    re.compile(r"\bconclude\b", re.IGNORECASE),
    re.compile(r"\brecommend\b", re.IGNORECASE),
    re.compile(r"\bpropose\b", re.IGNORECASE),
    re.compile(r"\bdocument\b", re.IGNORECASE),
    re.compile(r"\bvalidate\b", re.IGNORECASE),
    re.compile(r"\bdiagnose\b", re.IGNORECASE),
    re.compile(r"\bquery\b", re.IGNORECASE),
    re.compile(r"\bfind\b", re.IGNORECASE),
    re.compile(r"\blocate\b", re.IGNORECASE),
    re.compile(r"\bcollect\b", re.IGNORECASE),
    re.compile(r"\bgather\b", re.IGNORECASE),
    re.compile(r"\binterpret\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bpresent\b", re.IGNORECASE),
    re.compile(r"\bdeliver\b", re.IGNORECASE),
    re.compile(r"\bsubmit\b", re.IGNORECASE),
    re.compile(r"\bexecute\b", re.IGNORECASE),
    re.compile(r"\brun\b", re.IGNORECASE),
    re.compile(r"\bperform\b", re.IGNORECASE),
    re.compile(r"\bconduct\b", re.IGNORECASE),
    re.compile(r"\bexplore\b", re.IGNORECASE),
    re.compile(r"\bnavigate\b", re.IGNORECASE),
    re.compile(r"\bclick\b", re.IGNORECASE),
    re.compile(r"\btype\b", re.IGNORECASE),
    re.compile(r"\benter\b", re.IGNORECASE),
    re.compile(r"\bscroll\b", re.IGNORECASE),
    re.compile(r"\bdownload\b", re.IGNORECASE),
    re.compile(r"\bupload\b", re.IGNORECASE),
]

# Conjunction / clause separators
# Two-pass: first split on commas that introduce a new independent clause
# (followed by a capitalized word or a conjunction like "and/but/then"),
# then split on conjunction markers (kept in result).
_SEP_COMMA = re.compile(
    r"\s*,\s+(?=(?:and|but|then|also|finally|next|after|before|meanwhile)\s)",
    re.IGNORECASE,
)
# Conjunction split: match "and finally", "and then", "but then", etc.
# as a single separator token so it stays with the following clause.
_SEP_CONJ = re.compile(
    r"\s*(?:and\s+then|and\s+also|and\s+finally|but\s+then|"
    r"also\s+and|then|but|and|finally|also|"
    r"furthermore| additionally| next| thereafter| afterwards|"
    r"after\s+that| before\s+that| meanwhile| while| whereas)\s+",
    re.IGNORECASE,
)

# Sentence-ending split
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Milestone dataclass
# ---------------------------------------------------------------------------


@dataclass
class Milestone:
    """
    A single verifiable sub-goal extracted from a task description.

    Attributes
    ----------
    id : uuid.UUID
        Unique identifier for this milestone.
    description : str
        Human-readable description of what this milestone requires.
    evidence_spec : str
        Description of what evidence / artifacts are needed to verify
        this milestone was achieved.
    success_rubric : str
        Concrete criteria that determine whether this milestone passes.
    order : int
        Zero-based execution order (lower = earlier).
    """

    description: str
    evidence_spec: str
    success_rubric: str
    order: int = 0
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "id": str(self.id),
            "description": self.description,
            "evidence_spec": self.evidence_spec,
            "success_rubric": self.success_rubric,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Milestone:
        """Reconstruct a Milestone from a dictionary (e.g. loaded from JSON)."""
        return cls(
            id=uuid.UUID(data["id"]),
            description=data["description"],
            evidence_spec=data["evidence_spec"],
            success_rubric=data["success_rubric"],
            order=data["order"],
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Milestone(id={str(self.id)!r}, order={self.order}, "
            f"description={self.description!r})"
        )


# ---------------------------------------------------------------------------
# Default evidence / rubric generators
# ---------------------------------------------------------------------------

_DEFAULT_ACTION_EVIDENCE = {
    "analyze": "Analytical notes, extracted insights, or structured findings.",
    "retrieve": "Retrieved data items, query results, or fetched documents.",
    "compute": "Numerical results, calculations, or derived values.",
    "verify": "Verification output, check results, or confirmation records.",
    "synthesize": "Synthesized summary, combined report, or integrated findings.",
    "evaluate": "Evaluation scores, assessment results, or rating tables.",
    "compare": "Comparison table, delta analysis, or relative metrics.",
    "generate": "Generated artifact, output file, or produced content.",
    "select": "Selection rationale, chosen options, or decision records.",
    "decide": "Decision record, chosen option, or outcome justification.",
    "fetch": "Fetched data, API responses, or retrieved documents.",
    "obtain": "Obtained values, acquired data, or gathered evidence.",
    "identify": "Identified entities, named items, or discovered values.",
    "extract": "Extracted data, parsed content, or isolated information.",
    "build": "Built artifact, constructed object, or assembled output.",
    "create": "Created content, new document, or generated file.",
    "develop": "Developed plan, strategy, or structured approach.",
    "design": "Design specification, blueprint, or architectural plan.",
    "plan": "Planned steps, structured schedule, or action plan.",
    "prepare": "Prepared materials, ready resources, or organized content.",
    "review": "Reviewed items, annotated list, or assessment notes.",
    "assess": "Assessment results, evaluated scores, or risk ratings.",
    "check": "Check results, validation output, or confirmation list.",
    "ensure": "Confirmation of condition, guarantee statement, or check output.",
    "establish": "Established baseline, set standard, or defined reference.",
    "define": "Defined terms, specifications, or clear boundaries.",
    "measure": "Measured values, metrics, or quantitative results.",
    "calculate": "Calculated results, arithmetic output, or derived figures.",
    "derive": "Derived values, computed results, or deduced conclusions.",
    "plot": "Plot, chart, graph, or visual representation.",
    "chart": "Chart, graph, visual, or diagrammatic output.",
    "report": "Report, summary, documented findings, or written output.",
    "summarize": "Summary, concise overview, or distilled key points.",
    "conclude": "Conclusion statement, final verdict, or deduced outcome.",
    "recommend": "Recommendation list, suggested actions, or expert advice.",
    "propose": "Proposed plan, suggested approach, or recommended actions.",
    "document": "Documentation, written record, or annotated output.",
    "validate": "Validation results, confirmed correctness, or check output.",
    "diagnose": "Diagnosis, identified root cause, or problem analysis.",
    "query": "Query results, database output, or retrieved records.",
    "find": "Found items, located resources, or discovered data.",
    "locate": "Located items, identified positions, or found resources.",
    "collect": "Collected data, gathered samples, or aggregated records.",
    "gather": "Gathered information, collected data, or assembled inputs.",
    "interpret": "Interpretation, explained meaning, or clarified insights.",
    "explain": "Explanation, reasoning, or detailed description.",
    "present": "Presented results, delivered content, or shared findings.",
    "deliver": "Delivered output, submitted artifact, or completed work.",
    "submit": "Submitted content, completed form, or sent output.",
    "execute": "Executed actions, run commands, or completed steps.",
    "run": "Run results, execution output, or completed process.",
    "perform": "Performed actions, completed operations, or execution log.",
    "conduct": "Conducted analysis, performed study, or executed process.",
    "explore": "Explored options, investigated paths, or surveyed space.",
    "navigate": "Navigation path, visited locations, or UI state log.",
    "click": "Click actions, UI interactions, or interaction log.",
    "type": "Typed input, entered text, or form submission log.",
    "enter": "Entered data, filled fields, or submitted values.",
    "scroll": "Scrolled content, observed UI elements, or viewed data.",
    "download": "Downloaded file, saved artifact, or retrieved content.",
    "upload": "Uploaded file, submitted data, or sent artifact.",
}


def _make_evidence_spec(clause: str) -> str:
    """Derive an evidence_spec string from a clause."""
    clause_lower = clause.lower()
    for action, spec in _DEFAULT_ACTION_EVIDENCE.items():
        if action in clause_lower:
            return spec
    return (
        f"Output, artifact, or record demonstrating completion of: {clause.strip()}"
    )


def _make_success_rubric(description: str, order: int) -> str:
    """Derive a generic success rubric from a milestone description."""
    return (
        f"Milestone {order + 1} is considered PASS when: "
        f"the agent has demonstrably completed the described step "
        f"and produced the required evidence artifacts. "
        f"FAIL when: the step was not completed or evidence is missing. "
        f"UNCERTAIN when: evidence is ambiguous or incomplete."
    )


# ---------------------------------------------------------------------------
# Heuristic clause extraction
# ---------------------------------------------------------------------------


def _split_into_clauses(task: str) -> List[str]:
    """
    Split a task string into relatively independent clauses.

    Two-pass approach:
    1. Split on sentence boundaries.
    2. Within each segment, split on commas/semicolons as standalone delimiters.
    3. Then split on conjunction markers (and, but, then, etc.), keeping the
       conjunction word at the start of the new clause so it reads naturally.
    """
    # Normalise whitespace
    task = re.sub(r"\s+", " ", task).strip()

    # Split on sentence boundaries
    raw_segments = SENTENCE_SPLIT.split(task)
    clauses: List[str] = []

    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue

        # First pass: split on commas/semicolons that are standalone delimiters
        # (i.e. not part of a quoted string or similar)
        comma_parts: List[str] = []
        for part in _SEP_COMMA.split(seg):
            part = part.strip()
            if part:
                comma_parts.append(part)

        # Second pass: split each comma-part on conjunction markers
        for part in comma_parts:
            sub_parts = _SEP_CONJ.split(part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    clauses.append(sp)

    return clauses


def _identify_verb(clause: str) -> Optional[str]:
    """Return the first action-verb keyword found in the clause, or None."""
    clause_lower = clause.lower()
    for pattern in ACTION_PATTERNS:
        if pattern.search(clause_lower):
            match = pattern.search(clause_lower)
            if match:
                return match.group(0).lower()
    return None


def _clause_to_milestone(clause: str, order: int) -> Milestone:
    """Convert a single clause into a Milestone object."""
    verb = _identify_verb(clause)
    evidence_spec = _make_evidence_spec(clause)
    success_rubric = _make_success_rubric(clause, order)
    return Milestone(
        id=uuid.uuid4(),
        description=clause.strip(),
        evidence_spec=evidence_spec,
        success_rubric=success_rubric,
        order=order,
    )


# ---------------------------------------------------------------------------
# MilestoneDecomposer
# ---------------------------------------------------------------------------


class MilestoneDecomposer:
    """
    Decomposes a natural-language task description into a ranked list of
    verifiable :class:`Milestone` objects.

    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key. If ``None``, the ``OPENAI_API_KEY`` environment
        variable is consulted. If a key is available, the :meth:`refine`
        method will use ``gpt-4o-mini`` to improve milestone quality.
    model : str, optional
        OpenAI model to use for refinement.  Defaults to ``"gpt-4o-mini"``.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._client: Optional[OpenAI] = None
        if _OPENAI_AVAILABLE and self._api_key:
            self._client = OpenAI(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, task: str) -> List[Milestone]:
        """
        Decompose a task description into a list of :class:`Milestone` objects.

        This method applies a **heuristic decomposition** only:
        - Splits the task on sentence boundaries and conjunctions
        - Identifies action verbs in each clause
        - Generates evidence specifications and success rubrics
        - Assigns sequential ordering based on clause position

        Parameters
        ----------
        task : str
            The natural-language task to decompose.

        Returns
        -------
        List[Milestone]
            Ordered list of milestones (order 0 → N-1).

        Raises
        ------
        ValueError
            If ``task`` is empty or blank.
        """
        if not task or not task.strip():
            raise ValueError("Task description must not be empty.")

        clauses = _split_into_clauses(task.strip())

        if not clauses:
            # Fallback: treat the whole task as a single milestone
            clauses = [task.strip()]

        milestones: List[Milestone] = []
        for order, clause in enumerate(clauses):
            milestone = _clause_to_milestone(clause, order)
            milestones.append(milestone)

        return milestones

    def refine(
        self,
        milestones: List[Milestone],
        task: str,
    ) -> List[Milestone]:
        """
        Refine an existing list of milestones using an LLM.

        If ``OPENAI_API_KEY`` is set (or passed to the constructor),
        this method sends the current milestones and the original task
        to ``gpt-4o-mini`` and asks it to improve the milestone descriptions,
        evidence specs, and success rubrics.

        If no API key is available, the original milestones are returned
        unchanged (no-op).

        Parameters
        ----------
        milestones : List[Milestone]
            The milestones to refine.
        task : str
            The original task description (used as context for the LLM).

        Returns
        -------
        List[Milestone]
            Refined milestones, or the original list if no API key is set.
        """
        if not milestones:
            return milestones

        if not self._client:
            # No API key — return originals unchanged
            return list(milestones)

        # Build the prompt
        prompt = self._build_refinement_prompt(milestones, task)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at decomposing tasks into precise, "
                            "verifiable milestones. Given a task and a list of "
                            "heuristically-derived milestones, improve them: make "
                            "descriptions more precise, make evidence_spec "
                            "actionable, and make success_rubric concrete and "
                            "measurable. Return ONLY a valid JSON array of "
                            "milestone objects with keys: description, "
                            "evidence_spec, success_rubric, order. "
                            "Do not include any explanation outside the JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or "[]"

            # Strip markdown fences if present
            raw = re.sub(r"^```json\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())

            import json

            refined_data = json.loads(raw)

            # Rebuild milestones preserving original IDs where possible
            refined_milestones: List[Milestone] = []
            for item in refined_data:
                orig = next(
                    (m for m in milestones if m.order == item.get("order", -1)),
                    milestones[len(refined_milestones)]
                    if len(refined_milestones) < len(milestones)
                    else milestones[-1],
                )
                refined_milestones.append(
                    Milestone(
                        id=orig.id,
                        description=str(item.get("description", orig.description)),
                        evidence_spec=str(
                            item.get("evidence_spec", orig.evidence_spec)
                        ),
                        success_rubric=str(
                            item.get("success_rubric", orig.success_rubric)
                        ),
                        order=int(item.get("order", orig.order)),
                    )
                )

            # Re-sort by order
            refined_milestones.sort(key=lambda m: m.order)
            return refined_milestones

        except Exception:
            # On any LLM error, fall back to original milestones
            return list(milestones)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_refinement_prompt(
        milestones: List[Milestone], task: str
    ) -> str:
        """Build the user-prompt string for the refinement LLM call."""
        milestone_lines = []
        for m in milestones:
            milestone_lines.append(
                f'{{"order": {m.order}, '
                f'"description": {m.description!r}, '
                f'"evidence_spec": {m.evidence_spec!r}, '
                f'"success_rubric": {m.success_rubric!r}}}'
            )
        milestones_json = "[\n  " + ",\n  ".join(milestone_lines) + "\n]"
        return (
            f"Original task:\n{task}\n\n"
            f"Current milestones (may be imperfect):\n{milestones_json}\n\n"
            f"Improve these milestones. Return a JSON array. Each object "
            f"must have: description, evidence_spec, success_rubric, order."
        )
