"""
Milestone Critic — OS-Themis style per-milestone verdict + Box Maze boundary enforcement.

This module implements Step 4 of the HiMCA pipeline: given a list of milestones
and the evidence collected for each (from Step 3), render a pass/fail/uncertain
verdict for every milestone and check for boundary violations using the
BoxMazeBoundaryLayer.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.decompose import Milestone
from src.evidence import Evidence


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class MilestoneVerdict(Enum):
    """
    The critic's overall assessment for a single milestone.

    PASS   — Strong evidence confirms the milestone was achieved.
    FAIL   — Strong evidence confirms the milestone was NOT achieved.
    UNCERTAIN — Evidence is ambiguous, insufficient, or contradictory.
    """

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Boundary violation dataclass
# ---------------------------------------------------------------------------


@dataclass
class BoundaryViolation:
    """
    A single boundary violation detected by the BoxMazeBoundaryLayer.

    Attributes
    ----------
    violation_type : str
        Category of the violation, e.g. "constraint_overstep",
        "memory_grounding_failure", "out_of_bounds_action".
    description : str
        Human-readable description of the violation.
    severity : str
        One of "low", "medium", "high".
    milestone_id : str
        UUID string of the milestone where the violation was detected.
    """

    violation_type: str
    description: str
    severity: str = "medium"
    milestone_id: str = ""


# ---------------------------------------------------------------------------
# Critic result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MilestoneCriticResult:
    """
    The complete output of the MilestoneCritic for a single milestone.

    Attributes
    ----------
    milestone_id : str
        UUID string of the milestone this result belongs to.
    verdict : MilestoneVerdict
        Final PASS / FAIL / UNCERTAIN verdict.
    reasoning : str
        Free-text explanation of why this verdict was reached.
    confidence : float
        Confidence score in [0.0, 1.0] for the verdict itself
        (separate from evidence confidence).
    violations : List[BoundaryViolation]
        List of boundary violations detected at this milestone.
    evidence_quality : float
        Sub-score for evidence quality in [0.0, 1.0].
    evidence_sufficiency : float
        Sub-score for whether enough evidence was gathered in [0.0, 1.0].
    evidence_consistency : float
        Sub-score for whether evidence is internally consistent in [0.0, 1.0].
    """

    milestone_id: str
    verdict: MilestoneVerdict
    reasoning: str
    confidence: float = 0.0
    violations: List[BoundaryViolation] = field(default_factory=list)
    evidence_quality: float = 0.0
    evidence_sufficiency: float = 0.0
    evidence_consistency: float = 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "milestone_id": self.milestone_id,
            "verdict": self.verdict.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "violations": [
                {
                    "violation_type": v.violation_type,
                    "description": v.description,
                    "severity": v.severity,
                    "milestone_id": v.milestone_id,
                }
                for v in self.violations
            ],
            "evidence_quality": self.evidence_quality,
            "evidence_sufficiency": self.evidence_sufficiency,
            "evidence_consistency": self.evidence_consistency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MilestoneCriticResult:
        """Reconstruct from a dictionary."""
        violations = [
            BoundaryViolation(
                violation_type=str(v.get("violation_type", "")),
                description=str(v.get("description", "")),
                severity=str(v.get("severity", "medium")),
                milestone_id=str(v.get("milestone_id", "")),
            )
            for v in data.get("violations", [])
        ]
        return cls(
            milestone_id=str(data["milestone_id"]),
            verdict=MilestoneVerdict(data["verdict"]),
            reasoning=str(data["reasoning"]),
            confidence=float(data.get("confidence", 0.0)),
            violations=violations,
            evidence_quality=float(data.get("evidence_quality", 0.0)),
            evidence_sufficiency=float(data.get("evidence_sufficiency", 0.0)),
            evidence_consistency=float(data.get("evidence_consistency", 0.0)),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MilestoneCriticResult(milestone_id={self.milestone_id!r}, "
            f"verdict={self.verdict.value!r}, confidence={self.confidence:.2f}, "
            f"violations={len(self.violations)})"
        )


# ---------------------------------------------------------------------------
# BoxMazeBoundaryLayer
# ---------------------------------------------------------------------------


class BoxMazeBoundaryLayer:
    """
    Checks for boundary violations at each milestone step using Box Maze
    constraint semantics.

    The "Box Maze" model treats each milestone as a constrained region in a
    problem-space. Boundary violations occur when:
    1. A constraint declared in the milestone is violated by the evidence.
    2. A step in the trajectory goes outside the defined problem bounds.
    3. Memory grounding fails — the agent's internal state diverges from
       what the evidence claims.
    4. An action is taken that contradicts a previously established constraint.

    Parameters
    ----------
    strict_mode : bool
        If True, more violations are flagged. Defaults to False.
    max_violations_per_milestone : int
        Cap on violations stored per milestone. Defaults to 10.
    """

    # Patterns that indicate a constraint being stated in a milestone
    _CONSTRAINT_PATTERNS = [
        re.compile(r"\bnot\b", re.IGNORECASE),
        re.compile(r"\bexclude\b", re.IGNORECASE),
        re.compile(r"\bavoid\b", re.IGNORECASE),
        re.compile(r"\bonly\b", re.IGNORECASE),
        re.compile(r"\bmust\s+not\b", re.IGNORECASE),
        re.compile(r"\bshould\s+not\b", re.IGNORECASE),
        re.compile(r"\bnever\b", re.IGNORECASE),
        re.compile(r"\bno\s+more\s+than\b", re.IGNORECASE),
        re.compile(r"\blimit(?:ed)?\s+to\b", re.IGNORECASE),
        re.compile(r"\bgreater\s+than\b", re.IGNORECASE),
        re.compile(r"\bless\s+than\b", re.IGNORECASE),
        re.compile(r"\bwithin\b", re.IGNORECASE),
        re.compile(r"\bbetween\b", re.IGNORECASE),
        re.compile(r"\bmaximum\b", re.IGNORECASE),
        re.compile(r"\bminimum\b", re.IGNORECASE),
    ]

    # Patterns that suggest an out-of-bounds action was taken
    _OUT_OF_BOUNDS_PATTERNS = [
        re.compile(r"\bskip\b", re.IGNORECASE),
        re.compile(r"\bignore\b", re.IGNORECASE),
        re.compile(r"\bskip(?:ped)?\s+the\b", re.IGNORECASE),
        re.compile(r"\bno\s+(?:mention|evidence)\s+of\b", re.IGNORECASE),
        re.compile(r"\bfail(?:ed)?\s+to\b", re.IGNORECASE),
        re.compile(r"\bmiss(?:ing|ed)?\s+the\b", re.IGNORECASE),
        re.compile(r"\bcontrary\s+to\b", re.IGNORECASE),
        re.compile(r"\bviolate[ds]?\b", re.IGNORECASE),
        re.compile(r"\bbroke[n]?\s+the\b", re.IGNORECASE),
    ]

    # Patterns suggesting memory grounding failure
    _MEMORY_GROUNDING_PATTERNS = [
        re.compile(r"\bstated\s+that\b.*but\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bpreviously\s+(?:found|identified)\b.*now\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bhowever\b.*\bcontradict\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bshould\s+be\b.*\bbut\s+is\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bexpected\b.*\bactual\b", re.IGNORECASE | re.DOTALL),
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        max_violations_per_milestone: int = 10,
    ) -> None:
        self._strict_mode = strict_mode
        self._max_violations = max_violations_per_milestone
        # Track previously seen constraints across milestones for cross-check
        self._seen_constraints: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_boundary(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> List[BoundaryViolation]:
        """
        Check a single milestone for boundary violations.

        Parameters
        ----------
        milestone : Milestone
            The milestone to check.
        evidence : Evidence
            The evidence collected for this milestone.

        Returns
        -------
        List[BoundaryViolation]
            List of detected violations (may be empty).
        """
        violations: List[BoundaryViolation] = []

        # 1. Constraint pattern violations
        constraint_violations = self._check_constraints(milestone, evidence)
        violations.extend(constraint_violations)

        # 2. Out-of-bounds action violations
        oob_violations = self._check_out_of_bounds(milestone, evidence)
        violations.extend(oob_violations)

        # 3. Memory grounding failures
        grounding_violations = self._check_memory_grounding(milestone, evidence)
        violations.extend(grounding_violations)

        # 4. Inconsistency violations (evidence vs. milestone description)
        inconsistency_violations = self._check_inconsistency(milestone, evidence)
        violations.extend(inconsistency_violations)

        # 5. Evidence sufficiency violations (Box Maze style)
        sufficiency_violations = self._check_sufficiency(milestone, evidence)
        violations.extend(sufficiency_violations)

        # 6. In strict mode, flag low-confidence evidence as violations
        if self._strict_mode and evidence.confidence < 0.5:
            violations.append(
                BoundaryViolation(
                    violation_type="low_confidence_evidence",
                    description=(
                        f"Evidence confidence {evidence.confidence:.2f} is below "
                        f"0.5 threshold in strict mode"
                    ),
                    severity="low",
                    milestone_id=str(milestone.id),
                )
            )

        # Cap at max violations
        return violations[: self._max_violations]

    def reset_constraints(self) -> None:
        """Clear the constraint memory. Call between independent evaluation runs."""
        self._seen_constraints.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_constraints(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> List[BoundaryViolation]:
        """Detect constraint-overstep violations."""
        violations: List[BoundaryViolation] = []
        desc_lower = milestone.description.lower()

        # Identify constraint keywords in the milestone description
        constraint_keywords: List[str] = []
        for pattern in self._CONSTRAINT_PATTERNS:
            for match in pattern.finditer(desc_lower):
                constraint_keywords.append(match.group(0).lower())

        if not constraint_keywords:
            return violations

        # Build a combined text from all evidence snippets
        evidence_text = self._build_evidence_text(evidence)

        for kw in constraint_keywords:
            if kw in ["not", "only", "must not", "should not", "never"]:
                # Check if evidence shows the constraint was violated
                violation = self._check_constraint_violation(
                    kw, milestone, evidence, evidence_text
                )
                if violation:
                    violations.append(violation)

        return violations

    def _check_constraint_violation(
        self,
        constraint_kw: str,
        milestone: Milestone,
        evidence: Evidence,
        evidence_text: str,
    ) -> Optional[BoundaryViolation]:
        """Check if a specific constraint keyword was violated by the evidence."""
        negations = {"not", "must not", "should not", "never", "avoid", "exclude"}
        positive_constraints = {"only", "limit", "within", "between", "maximum", "minimum"}

        if constraint_kw in negations:
            # Look for evidence that violates a "must not" constraint
            # Heuristic: if the milestone says "not X" but evidence mentions "X"
            # we flag a violation
            action_verb = self._extract_action_verb(milestone.description)
            if action_verb and action_verb in evidence_text.lower():
                # The action we were told NOT to do appears in evidence
                return BoundaryViolation(
                    violation_type="constraint_overstep",
                    description=(
                        f"Constraint '{constraint_kw}' appears to be violated: "
                        f"the evidence mentions the action '{action_verb}' which was "
                        f"subject to a '{constraint_kw}' constraint"
                    ),
                    severity="high",
                    milestone_id=str(milestone.id),
                )

        if constraint_kw in positive_constraints:
            # Check bounds violations for "only", "limit to", etc.
            pass  # Basic implementation; extend as needed

        return None

    def _check_out_of_bounds(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> List[BoundaryViolation]:
        """Detect out-of-bounds action violations."""
        violations: List[BoundaryViolation] = []
        evidence_text = self._build_evidence_text(evidence)

        for pattern in self._OUT_OF_BOUNDS_PATTERNS:
            matches = list(pattern.finditer(evidence_text))
            for match in matches:
                violations.append(
                    BoundaryViolation(
                        violation_type="out_of_bounds_action",
                        description=(
                            f"Out-of-bounds indicator detected: '{match.group(0)}'. "
                            f"This suggests an action was skipped, ignored, or "
                            f"contradicted a boundary condition."
                        ),
                        severity="medium",
                        milestone_id=str(milestone.id),
                    )
                )

        return violations

    def _check_memory_grounding(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> List[BoundaryViolation]:
        """Detect memory grounding failures."""
        violations: List[BoundaryViolation] = []
        evidence_text = self._build_evidence_text(evidence)

        for pattern in self._MEMORY_GROUNDING_PATTERNS:
            matches = list(pattern.finditer(evidence_text))
            for match in matches:
                violations.append(
                    BoundaryViolation(
                        violation_type="memory_grounding_failure",
                        description=(
                            f"Memory grounding failure detected near: "
                            f"'{match.group(0)[:80]}...'. The agent's internal state "
                            f"appears to contradict previously established facts."
                        ),
                        severity="high",
                        milestone_id=str(milestone.id),
                    )
                )

        return violations

    def _check_inconsistency(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> List[BoundaryViolation]:
        """Detect inconsistencies between milestone description and evidence."""
        violations: List[BoundaryViolation] = []

        # Check for numeric inconsistencies
        milestone_numbers = self._extract_numbers(milestone.description)
        evidence_numbers = self._extract_numbers(
            self._build_evidence_text(evidence)
        )

        for num_m in milestone_numbers:
            for num_e in evidence_numbers:
                # Very loose check: if milestone mentions a number range/bound
                # and evidence shows something wildly different, flag it
                # This is a simplified heuristic
                pass  # Keep basic; real implementation would be more sophisticated

        # Check citation count: zero citations with high evidence text count
        # suggests the evidence might be fabricated
        if (
            len(evidence.citations) == 0
            and len(evidence.text_snippets) >= 5
            and evidence.confidence > 0.7
        ):
            violations.append(
                BoundaryViolation(
                    violation_type="evidence_inconsistency",
                    description=(
                        f"High confidence ({evidence.confidence:.2f}) with zero "
                        f"citations suggests potential evidence inconsistency or "
                        f"fabrication."
                    ),
                    severity="medium",
                    milestone_id=str(milestone.id),
                )
            )

        # Check for contradictory intermediate results
        if len(evidence.intermediate_results) >= 2:
            if self._has_contradictory_results(evidence.intermediate_results):
                violations.append(
                    BoundaryViolation(
                        violation_type="contradictory_intermediate_results",
                        description=(
                            "Multiple intermediate results appear to contradict "
                            "each other."
                        ),
                        severity="high",
                        milestone_id=str(milestone.id),
                    )
                )

        return violations

    def _check_sufficiency(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> List[BoundaryViolation]:
        """Detect evidence sufficiency violations (Box Maze style)."""
        violations: List[BoundaryViolation] = []

        # Check if evidence_spec expectations were met
        spec_keywords = self._extract_keywords(milestone.evidence_spec)
        evidence_text = self._build_evidence_text(evidence)
        matched_keywords = [
            kw for kw in spec_keywords
            if len(kw) >= 4 and re.search(rf"\b{re.escape(kw)}\b", evidence_text)
        ]

        if len(spec_keywords) > 0:
            coverage = len(matched_keywords) / len(spec_keywords)
            if coverage < 0.3:
                violations.append(
                    BoundaryViolation(
                        violation_type="insufficient_evidence",
                        description=(
                            f"Evidence keyword coverage is only {coverage:.0%}. "
                            f"Expected keywords from evidence_spec: {spec_keywords}"
                        ),
                        severity="medium",
                        milestone_id=str(milestone.id),
                    )
                )

        return violations

    @staticmethod
    def _build_evidence_text(evidence: Evidence) -> str:
        """Combine all evidence text into a single string for pattern matching."""
        parts: List[str] = list(evidence.text_snippets)
        for citation in evidence.citations:
            parts.append(str(citation.get("text", "")))
        for result in evidence.intermediate_results:
            parts.append(str(result))
        return " ".join(parts)

    @staticmethod
    def _extract_action_verb(text: str) -> Optional[str]:
        """Extract the primary action verb from text."""
        action_patterns = [
            r"\banalyze[sd]?\b",
            r"\bretrieve[sd]?\b",
            r"\bcompute[sd]?\b",
            r"\bverify[sd]?\b",
            r"\bsynthesize[sd]?\b",
            r"\bevaluate[sd]?\b",
            r"\bcompare[sd]?\b",
            r"\bgenerate[sd]?\b",
            r"\bselect[ed]?\b",
            r"\bdecide[sd]?\b",
            r"\bfetch[ed]?\b",
            r"\bobtain[ed]?\b",
            r"\bidentify[ed]?\b",
            r"\bextract[ed]?\b",
            r"\bbuild[ed]?\b",
            r"\bcreate[ed]?\b",
            r"\bcheck[ed]?\b",
            r"\bmeasure[ed]?\b",
            r"\bcalculate[ed]?\b",
            r"\bquery[ed]?\b",
            r"\bfind[ing]?\b",
            r"\blocate[ed]?\b",
            r"\bcollect[ed]?\b",
            r"\binterpret[ed]?\b",
            r"\bvalidate[ed]?\b",
            r"\breport[ed]?\b",
            r"\bsummarize[ed]?\b",
            r"\bconclude[ed]?\b",
        ]
        for pattern in action_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(0).lower()
        return None

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Extract decimal/integer numbers from text."""
        return [
            float(m.group(1))
            for m in re.finditer(r"(?<![a-zA-Z0-9])(-?\d+\.?\d*)", text)
        ]

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract significant keywords (tokens ≥3 chars) from text."""
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "her", "was", "one", "our", "out", "day", "get",
            "has", "him", "his", "how", "its", "may", "new", "now",
            "old", "see", "two", "who", "boy", "did", "she", "use",
            "way", "will", "with", "from", "this", "that", "have",
            "more", "then", "into", "some", "would", "make", "like",
            "just", "over", "such", "take", "only", "come", "these",
            "could", "first", "after", "most", "also", "back",
        }
        return [t for t in tokens if t not in stop_words]

    @staticmethod
    def _has_contradictory_results(results: List[Any]) -> bool:
        """Check if a list of intermediate results contains contradictions."""
        # Simple heuristic: look for opposing boolean signals
        true_vals = {"true", "yes", "pass", "success", "correct", "valid", "confirmed"}
        false_vals = {"false", "no", "fail", "failure", "incorrect", "invalid", "rejected"}
        str_results = [str(r).lower().strip() for r in results]
        has_true = any(any(tv in s for tv in true_vals) for s in str_results)
        has_false = any(any(fv in s for fv in false_vals) for s in str_results)
        return has_true and has_false


# ---------------------------------------------------------------------------
# MilestoneCritic
# ---------------------------------------------------------------------------


class MilestoneCritic:
    """
    OS-Themis style milestone critic.

    Evaluates each milestone against its collected evidence and renders a
    PASS / FAIL / UNCERTAIN verdict. Integrates the BoxMazeBoundaryLayer
    for constraint checking.

    The OS-Themis style assessment considers three dimensions:
    1. **Evidence quality** — Are the text snippets and citations relevant
       and specific?
    2. **Evidence sufficiency** — Is there enough evidence to support a
       confident verdict?
    3. **Evidence consistency** — Do all pieces of evidence agree with
       each other and with the milestone description?

    Parameters
    ----------
    boundary_layer : BoxMazeBoundaryLayer, optional
        The boundary enforcement layer. If None, a default one is created.
    confidence_threshold_pass : float
        Minimum combined score to render a PASS verdict.
        Defaults to 0.7.
    confidence_threshold_fail : float
        Maximum combined score to render a FAIL verdict.
        Defaults to 0.3. Scores between fail and pass thresholds
        result in UNCERTAIN.
    require_violations_for_fail : bool
        If True, FAIL verdicts require at least one boundary violation.
        Defaults to False.

    Examples
    --------
    >>> from src.decompose import MilestoneDecomposer
    >>> from src.evidence import EvidenceCollector
    >>> from src.critic import MilestoneCritic
    >>> decomposer = MilestoneDecomposer()
    >>> collector = EvidenceCollector()
    >>> critic = MilestoneCritic()
    >>> milestones = decomposer.decompose("Analyze revenue and generate a report")
    >>> trajectory = [
    ...     {"action": "analyze revenue", "observation": "Revenue = $1.2M", "thought": "...", "timestamp": "2024-01-01T00:00:00Z"},
    ...     {"action": "generate PDF report", "observation": "Report saved to /tmp/report.pdf", "thought": "...", "timestamp": "2024-01-01T00:01:00Z"},
    ... ]
    >>> evidence_list = [collector.collect(m, trajectory) for m in milestones]
    >>> results = critic.critique(milestones, evidence_list)
    >>> for r in results:
    ...     print(r.verdict.value, r.confidence)
    """

    def __init__(
        self,
        boundary_layer: Optional[BoxMazeBoundaryLayer] = None,
        confidence_threshold_pass: float = 0.7,
        confidence_threshold_fail: float = 0.3,
        require_violations_for_fail: bool = False,
    ) -> None:
        if not (0.0 <= confidence_threshold_fail <= 1.0):
            raise ValueError("confidence_threshold_fail must be in [0.0, 1.0]")
        if not (0.0 <= confidence_threshold_pass <= 1.0):
            raise ValueError("confidence_threshold_pass must be in [0.0, 1.0]")
        if confidence_threshold_pass <= confidence_threshold_fail:
            raise ValueError(
                "confidence_threshold_pass must be > confidence_threshold_fail"
            )

        self._boundary = boundary_layer or BoxMazeBoundaryLayer()
        self._pass_threshold = confidence_threshold_pass
        self._fail_threshold = confidence_threshold_fail
        self._require_violations_for_fail = require_violations_for_fail

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def critique(
        self,
        milestones: List[Milestone],
        evidence_list: List[Evidence],
    ) -> List[MilestoneCriticResult]:
        """
        Evaluate milestones against their collected evidence.

        Parameters
        ----------
        milestones : List[Milestone]
            The milestones to evaluate (from Step 2).
        evidence_list : List[Evidence]
            The evidence collected for each milestone (from Step 3).
            Must be the same length as ``milestones`` and aligned by index.

        Returns
        -------
        List[MilestoneCriticResult]
            One result per milestone, in the same order as the input.

        Raises
        ------
        ValueError
            If the lengths of ``milestones`` and ``evidence_list`` do not match.
        """
        if len(milestones) != len(evidence_list):
            raise ValueError(
                f"milestones ({len(milestones)}) and evidence_list "
                f"({len(evidence_list)}) must have the same length"
            )

        results: List[MilestoneCriticResult] = []

        for milestone, evidence in zip(milestones, evidence_list):
            result = self._critique_single(milestone, evidence)
            results.append(result)

        return results

    def _critique_single(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> MilestoneCriticResult:
        """Evaluate a single milestone against its evidence."""
        # Step 1: Run boundary check
        violations = self._boundary.check_boundary(milestone, evidence)

        # Step 2: Compute sub-scores
        eq = self._score_evidence_quality(milestone, evidence)
        es = self._score_evidence_sufficiency(milestone, evidence)
        ec = self._score_evidence_consistency(milestone, evidence)

        # Step 3: Combine into an overall evidence score
        # Weighted average; evidence confidence is a prior from Step 3
        combined = (
            0.25 * evidence.confidence
            + 0.25 * eq
            + 0.25 * es
            + 0.25 * ec
        )

        # Step 4: Render verdict
        verdict, reasoning = self._determine_verdict(
            milestone, evidence, combined, violations, eq, es, ec
        )

        # Step 5: Compute confidence in the verdict itself
        verdict_confidence = self._score_verdict_confidence(
            combined, evidence, violations
        )

        return MilestoneCriticResult(
            milestone_id=str(milestone.id),
            verdict=verdict,
            reasoning=reasoning,
            confidence=verdict_confidence,
            violations=violations,
            evidence_quality=eq,
            evidence_sufficiency=es,
            evidence_consistency=ec,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_evidence_quality(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> float:
        """
        Score evidence quality — relevance and specificity of snippets/citations.

        Returns a score in [0.0, 1.0].
        """
        if not evidence.text_snippets and not evidence.citations:
            return 0.0

        score = 0.0
        milestone_keywords = BoxMazeBoundaryLayer._extract_keywords(
            milestone.description
        )

        # 1. Citation quality: do citations reference concrete sources?
        if evidence.citations:
            cited_snippets = [
                c["text"] for c in evidence.citations
                if c.get("text") and len(c["text"]) > 10
            ]
            score += min(0.4, len(cited_snippets) * 0.1)

        # 2. Snippet relevance: do snippets contain milestone keywords?
        relevant_snippets = 0
        for snippet in evidence.text_snippets:
            snippet_lower = snippet.lower()
            if any(
                len(kw) >= 4 and re.search(rf"\b{re.escape(kw)}\b", snippet_lower)
                for kw in milestone_keywords
            ):
                relevant_snippets += 1

        if evidence.text_snippets:
            relevance_ratio = relevant_snippets / len(evidence.text_snippets)
            score += relevance_ratio * 0.4

        # 3. Intermediate results add quality
        if evidence.intermediate_results:
            score += min(0.2, len(evidence.intermediate_results) * 0.05)

        return min(1.0, score)

    def _score_evidence_sufficiency(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> float:
        """
        Score evidence sufficiency — whether enough evidence was gathered.

        Returns a score in [0.0, 1.0].
        """
        # No evidence at all
        total_pieces = (
            len(evidence.text_snippets)
            + len(evidence.citations)
            + len(evidence.intermediate_results)
        )

        if total_pieces == 0:
            return 0.0

        # Penalise if evidence is very short
        total_chars = sum(len(s) for s in evidence.text_snippets)
        if total_chars < 20:
            char_penalty = 0.2
        elif total_chars < 50:
            char_penalty = 0.1
        else:
            char_penalty = 0.0

        # Reward more pieces up to a point
        volume_score = min(1.0, total_pieces / 5.0)

        # Confidence from Step 3 is a strong signal
        conf_signal = evidence.confidence

        combined = (volume_score * 0.4 + conf_signal * 0.6) - char_penalty
        return max(0.0, min(1.0, combined))

    def _score_evidence_consistency(
        self,
        milestone: Milestone,
        evidence: Evidence,
    ) -> float:
        """
        Score evidence consistency — do evidence pieces agree with each other
        and with the milestone?

        Returns a score in [0.0, 1.0].
        """
        # Zero or one piece of evidence = vacuously consistent
        total_pieces = (
            len(evidence.text_snippets)
            + len(evidence.citations)
            + len(evidence.intermediate_results)
        )

        if total_pieces <= 1:
            return 1.0

        consistency_score = 1.0

        # Check for contradictory intermediate results
        if self._boundary._has_contradictory_results(evidence.intermediate_results):
            consistency_score -= 0.5

        # Check if text snippets contradict milestone constraints
        desc_lower = milestone.description.lower()
        constraint_patterns = [
            re.compile(r"\bnot\b", re.IGNORECASE),
            re.compile(r"\bavoid\b", re.IGNORECASE),
            re.compile(r"\bexclude\b", re.IGNORECASE),
            re.compile(r"\bnever\b", re.IGNORECASE),
        ]
        has_constraint = any(p.search(desc_lower) for p in constraint_patterns)

        if has_constraint:
            # Look for violation signals in snippets
            violation_signals = [
                re.compile(rf"\b{constraint}\b", re.IGNORECASE)
                for constraint in ["skipped", "ignored", "failed", "missed"]
            ]
            for snippet in evidence.text_snippets:
                for vp in violation_signals:
                    if vp.search(snippet):
                        consistency_score -= 0.3
                        break

        return max(0.0, min(1.0, consistency_score))

    def _determine_verdict(
        self,
        milestone: Milestone,
        evidence: Evidence,
        combined_score: float,
        violations: List[BoundaryViolation],
        eq: float,
        es: float,
        ec: float,
    ) -> tuple[MilestoneVerdict, str]:
        """
        Determine the final verdict and reasoning text.

        Returns
        -------
        verdict : MilestoneVerdict
        reasoning : str
        """
        high_severity_violations = [
            v for v in violations if v.severity == "high"
        ]
        medium_violations = [
            v for v in violations if v.severity == "medium"
        ]

        # High-severity violations → FAIL (regardless of evidence score)
        if high_severity_violations:
            reasoning = (
                f"FAIL: {len(high_severity_violations)} high-severity boundary "
                f"violation(s) detected. "
                f"Evidence quality={eq:.2f}, sufficiency={es:.2f}, "
                f"consistency={ec:.2f}. "
                f"First violation: {high_severity_violations[0].description}"
            )
            return MilestoneVerdict.FAIL, reasoning

        # Require violations for FAIL
        if self._require_violations_for_fail:
            if violations:
                reasoning = (
                    f"FAIL: Boundary violation(s) detected. "
                    f"Evidence quality={eq:.2f}, sufficiency={es:.2f}, "
                    f"consistency={ec:.2f}. "
                    f"Combined score={combined_score:.2f}."
                )
                return MilestoneVerdict.FAIL, reasoning
            # No violations but require_violations_for_fail is True
            # Fall through to score-based verdict

        # Score-based verdict
        if combined_score >= self._pass_threshold:
            if medium_violations:
                # Medium violations → still PASS but note them
                reasoning = (
                    f"PASS (with {len(medium_violations)} medium-severity note(s)): "
                    f"Evidence quality={eq:.2f}, sufficiency={es:.2f}, "
                    f"consistency={ec:.2f}. Combined score={combined_score:.2f} "
                    f"exceeds threshold {self._pass_threshold:.2f}."
                )
            else:
                reasoning = (
                    f"PASS: Strong evidence. Quality={eq:.2f}, "
                    f"sufficiency={es:.2f}, consistency={ec:.2f}. "
                    f"Combined score={combined_score:.2f} >= "
                    f"threshold {self._pass_threshold:.2f}."
                )
            return MilestoneVerdict.PASS, reasoning

        if combined_score <= self._fail_threshold:
            reasoning = (
                f"FAIL: Insufficient or poor-quality evidence. "
                f"Quality={eq:.2f}, sufficiency={es:.2f}, "
                f"consistency={ec:.2f}. Combined score={combined_score:.2f} "
                f"<= threshold {self._fail_threshold:.2f}."
            )
            return MilestoneVerdict.FAIL, reasoning

        # Between thresholds → UNCERTAIN
        reasoning = (
            f"UNCERTAIN: Evidence is ambiguous or mixed. "
            f"Quality={eq:.2f}, sufficiency={es:.2f}, consistency={ec:.2f}. "
            f"Combined score={combined_score:.2f} lies between "
            f"pass ({self._pass_threshold:.2f}) and fail ({self._fail_threshold:.2f}) "
            f"thresholds."
        )
        return MilestoneVerdict.UNCERTAIN, reasoning

    def _score_verdict_confidence(
        self,
        combined_score: float,
        evidence: Evidence,
        violations: List[BoundaryViolation],
    ) -> float:
        """
        Score how confident we are in the verdict itself.

        Returns a score in [0.0, 1.0].
        """
        # Higher evidence confidence → more confident verdict
        base = evidence.confidence

        # More violations → less confidence
        violation_penalty = len(violations) * 0.05

        # If score is very close to a threshold boundary, reduce confidence
        margin_penalty = 0.0
        distance_to_nearest_threshold = min(
            abs(combined_score - self._pass_threshold),
            abs(combined_score - self._fail_threshold),
        )
        if distance_to_nearest_threshold < 0.1:
            margin_penalty = 0.1

        confidence = base - violation_penalty - margin_penalty
        return max(0.0, min(1.0, confidence))
