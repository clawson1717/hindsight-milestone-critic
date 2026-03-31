"""
Hindsight Experience Store — HeRL-style failed milestone storage and retrieval.

This module implements Step 5 of the HiMCA pipeline: given a completed episode
(milestones + evidence + critic verdicts + trajectory), store failed milestones
(FAIL or UNCERTAIN) in a hindsight experience buffer and provide efficient
retrieval for new tasks.

HeRL-style hindsight relabeling: when a milestone fails, store it with a
"what if this was the actual goal?" label so the policy can learn from failure.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.critic import MilestoneCriticResult, MilestoneVerdict
from src.decompose import Milestone
from src.evidence import Evidence

# ---------------------------------------------------------------------------
# HindsightEntry
# ---------------------------------------------------------------------------


@dataclass
class HindsightEntry:
    """
    A single stored experience from a failed milestone episode.

    Attributes
    ----------
    entry_id : str
        Unique identifier (UUID string) for this entry.
    original_task : str
        The full task description that led to this episode.
    failed_milestone : Milestone
        The milestone that received a FAIL or UNCERTAIN verdict.
    failed_milestone_index : int
        Zero-based index of the failed milestone in the milestone list.
    collected_evidence : Evidence
        The evidence collected for the failed milestone (from Step 3).
    critic_result : MilestoneCriticResult
        The critic's verdict and reasoning (from Step 4).
    trajectory : List[Dict[str, Any]]
        Full trajectory steps for this episode.
    hindsight_label : str
        HeRL-style relabeling: "What if [milestone description] was the actual goal?"
    retrieval_keywords : List[str]
        Pre-computed keywords for fast TF-IDF retrieval.
    keyword_tfidf : Dict[str, float]
        Pre-computed TF-IDF weights for retrieval.
    created_at : datetime
        When this entry was created (UTC).
    episode_id : str
        Groups entries from the same episode run.
    """

    original_task: str
    failed_milestone: Milestone
    failed_milestone_index: int
    collected_evidence: Evidence
    critic_result: MilestoneCriticResult
    trajectory: List[Dict[str, Any]]
    hindsight_label: str
    retrieval_keywords: List[str] = field(default_factory=list)
    keyword_tfidf: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "entry_id": self.entry_id,
            "original_task": self.original_task,
            "failed_milestone": self.failed_milestone.to_dict(),
            "failed_milestone_index": self.failed_milestone_index,
            "collected_evidence": self.collected_evidence.to_dict(),
            "critic_result": self.critic_result.to_dict(),
            "trajectory": self.trajectory,
            "hindsight_label": self.hindsight_label,
            "retrieval_keywords": self.retrieval_keywords,
            "keyword_tfidf": self.keyword_tfidf,
            "created_at": self.created_at.isoformat(),
            "episode_id": self.episode_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HindsightEntry:
        """Reconstruct from a JSON dictionary."""
        return cls(
            entry_id=str(data["entry_id"]),
            original_task=str(data["original_task"]),
            failed_milestone=Milestone.from_dict(data["failed_milestone"]),
            failed_milestone_index=int(data["failed_milestone_index"]),
            collected_evidence=Evidence.from_dict(data["collected_evidence"]),
            critic_result=MilestoneCriticResult.from_dict(data["critic_result"]),
            trajectory=list(data["trajectory"]),
            hindsight_label=str(data["hindsight_label"]),
            retrieval_keywords=list(data.get("retrieval_keywords", [])),
            keyword_tfidf=dict(data.get("keyword_tfidf", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            episode_id=str(data.get("episode_id", str(uuid.uuid4()))),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HindsightEntry(entry_id={self.entry_id[:8]!r}, "
            f"task={self.original_task[:40]!r}, "
            f"hindsight_label={self.hindsight_label[:40]!r})"
        )


# ---------------------------------------------------------------------------
# Keyword/TF-IDF utilities
# ---------------------------------------------------------------------------

# Stop words to exclude from keyword extraction
_STOP_WORDS: Set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "its", "may", "new", "now", "old", "see", "two", "who", "boy", "did",
    "she", "use", "way", "will", "with", "from", "this", "that", "have",
    "more", "then", "into", "some", "would", "make", "like", "just", "over",
    "such", "take", "only", "come", "these", "could", "first", "after",
    "most", "also", "back", "than", "them", "same", "well", "about",
    "being", "very", "your", "what", "when", "where", "which", "while",
    "who", "why", "how", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than", "too",
    "very", "can", "just", "should", "now", "because", "while", "if",
    "or", "and", "nor", "but", "yet", "so", "either", "neither",
    "not", "no", "yes", "also", "here", "there", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "once",
}


def _extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """Extract significant keywords (tokens ≥ min_len chars, not stop words)."""
    tokens = re.findall(r"\b[a-zA-Z]{" + str(min_len) + r",}\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _compute_tfidf(
    documents: List[str],
    vocab: Optional[Set[str]] = None,
) -> List[Dict[str, float]]:
    """
    Compute TF-IDF weights for a list of documents.

    Parameters
    ----------
    documents : List[str]
        List of document texts.
    vocab : Set[str], optional
        Pre-computed vocabulary (all keywords across all docs).

    Returns
    -------
    List[Dict[str, float]]
        One TF-IDF dict per document.
    """
    if not documents:
        return []

    # Tokenize each document
    tokenized: List[Counter] = []
    all_tokens: Set[str] = set(vocab or set())

    for doc in documents:
        tokens = _extract_keywords(doc)
        counter = Counter(tokens)
        tokenized.append(counter)
        all_tokens.update(tokens)

    if not all_tokens:
        return [{} for _ in documents]

    n_docs = len(documents)
    idf: Dict[str, float] = {}
    for term in all_tokens:
        doc_freq = sum(1 for counter in tokenized if counter.get(term, 0) > 0)
        idf[term] = math.log((n_docs + 1) / (doc_freq + 1)) + 1

    # Compute TF-IDF per document
    result: List[Dict[str, float]] = []
    for counter in tokenized:
        tfidf: Dict[str, float] = {}
        max_tf = max(counter.values()) if counter else 1
        for term, tf in counter.items():
            tf_norm = tf / max_tf
            tfidf[term] = tf_norm * idf[term]

        # Normalise to unit vector
        magnitude = math.sqrt(sum(v * v for v in tfidf.values()))
        if magnitude > 0:
            tfidf = {t: v / magnitude for t, v in tfidf.items()}

        result.append(tfidf)

    return result


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot_product = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def _keyword_overlap_score(keywords_a: List[str], keywords_b: List[str]) -> float:
    """Compute Jaccard-like keyword overlap between two keyword lists."""
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = set(keywords_a)
    set_b = set(keywords_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# HindsightStore
# ---------------------------------------------------------------------------


class HindsightStore:
    """
    HeRL-style hindsight experience store for failed milestones.

    Stores entries when a milestone receives FAIL or UNCERTAIN verdict.
    Provides TF-IDF and keyword-overlap retrieval for new tasks.

    Parameters
    ----------
    store_dir : pathlib.Path, optional
        Directory for JSON persistence. Defaults to ``./hindsight_store/``.
    max_entries : int, optional
        Maximum entries to store in memory (FIFO eviction). 0 = unlimited.
        Defaults to 0 (unlimited).

    Examples
    --------
    >>> from src.decompose import MilestoneDecomposer
    >>> from src.evidence import EvidenceCollector
    >>> from src.critic import MilestoneCritic
    >>> from src.hindsight import HindsightStore
    >>> store = HindsightStore()
    >>> decomposer = MilestoneDecomposer()
    >>> milestones = decomposer.decompose("Analyze revenue and generate a report")
    >>> trajectory = [...]
    >>> evidence_list = [collector.collect(m, trajectory) for m in milestones]
    >>> critic_results = critic.critique(milestones, evidence_list)
    >>> # Add failed milestones to store
    >>> entry_ids = store.add("Analyze revenue and generate a report",
    ...                        milestones, evidence_list, critic_results, trajectory)
    >>> # Retrieve similar failures for a new task
    >>> similar = store.retrieve("Analyze costs and generate a report", top_k=3)
    """

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        max_entries: int = 0,
    ) -> None:
        self._entries: List[HindsightEntry] = []
        self._max_entries = max_entries
        self._store_dir = Path(store_dir) if store_dir else Path("./hindsight_store")
        self._next_entry_idx = 0  # For FIFO eviction

        # Ensure store directory exists
        if self._store_dir:
            self._store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        task: str,
        milestones: List[Milestone],
        evidence_list: List[Evidence],
        critic_results: List[MilestoneCriticResult],
        trajectory: List[Dict[str, Any]],
        episode_id: Optional[str] = None,
    ) -> List[str]:
        """
        Add a completed episode, storing only failed milestones.

        Parameters
        ----------
        task : str
            The original task description.
        milestones : List[Milestone]
            All milestones for this task.
        evidence_list : List[Evidence]
            Evidence collected for each milestone (aligned by index).
        critic_results : List[MilestoneCriticResult]
            Critic verdicts for each milestone (aligned by index).
        trajectory : List[Dict[str, Any]]
            Full trajectory steps for this episode.
        episode_id : str, optional
            Grouping ID for entries from the same episode.
            Defaults to a new UUID.

        Returns
        -------
        List[str]
            List of entry IDs for the stored hindsight entries
            (one per failed milestone).
        """
        if len(milestones) != len(evidence_list) or len(milestones) != len(critic_results):
            raise ValueError(
                f"milestones ({len(milestones)}), evidence_list ({len(evidence_list)}), "
                f"and critic_results ({len(critic_results)}) must have the same length"
            )

        ep_id = episode_id or str(uuid.uuid4())
        new_entries: List[HindsightEntry] = []

        for idx, (milestone, evidence, result) in enumerate(
            zip(milestones, evidence_list, critic_results)
        ):
            # Only store failed or uncertain milestones
            if result.verdict not in (MilestoneVerdict.FAIL, MilestoneVerdict.UNCERTAIN):
                continue

            # Generate hindsight label (HeRL-style relabeling)
            hindsight_label = self._make_hindsight_label(milestone, result)

            # Build retrieval keywords from task + milestone + evidence
            combined_text = (
                f"{task} {milestone.description} {milestone.success_rubric} "
                f"{' '.join(evidence.text_snippets)}"
            )
            keywords = _extract_keywords(combined_text)
            keywords = list(set(keywords))  # Deduplicate

            entry = HindsightEntry(
                entry_id=str(uuid.uuid4()),
                original_task=task,
                failed_milestone=milestone,
                failed_milestone_index=idx,
                collected_evidence=evidence,
                critic_result=result,
                trajectory=list(trajectory),
                hindsight_label=hindsight_label,
                retrieval_keywords=keywords,
                keyword_tfidf={},
                created_at=datetime.now(timezone.utc),
                episode_id=ep_id,
            )

            new_entries.append(entry)

        if not new_entries:
            return []

        # Compute TF-IDF for all entries (including new ones)
        all_texts = [e.original_task + " " + e.failed_milestone.description for e in self._entries + new_entries]
        tfidf_vecs = _compute_tfidf(all_texts)

        n_existing = len(self._entries)
        for i, entry in enumerate(new_entries):
            if n_existing + i < len(tfidf_vecs):
                entry.keyword_tfidf = tfidf_vecs[n_existing + i]
            else:
                # Fallback: single-doc TF-IDF
                entry.keyword_tfidf = _compute_tfidf(
                    [entry.original_task + " " + entry.failed_milestone.description]
                )[0]

        self._entries.extend(new_entries)

        # FIFO eviction if max_entries is set
        if self._max_entries > 0:
            while len(self._entries) > self._max_entries:
                self._entries.pop(0)

        return [e.entry_id for e in new_entries]

    def retrieve(
        self,
        task: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        rubric_similarity_threshold: float = 0.0,
    ) -> List[HindsightEntry]:
        """
        Retrieve the top-k most similar failed experiences for a task.

        Uses a combination of:
        1. TF-IDF cosine similarity on (task + milestone description)
        2. Keyword Jaccard overlap on retrieval keywords

        Parameters
        ----------
        task : str
            The new task description.
        top_k : int
            Maximum number of entries to return. Defaults to 5.
        similarity_threshold : float
            Minimum combined similarity score (0.0–1.0) to include an entry.
            Defaults to 0.0 (return all up to top_k).
        rubric_similarity_threshold : float
            Minimum rubric-level similarity to include an entry.
            Defaults to 0.0.

        Returns
        -------
        List[HindsightEntry]
            Up to ``top_k`` entries sorted by combined similarity score,
            highest first. Returns empty list if store is empty or no
            entries meet the threshold.
        """
        if not self._entries or top_k <= 0:
            return []

        # Compute query TF-IDF
        task_keywords = _extract_keywords(task)
        query_text = task
        query_tfidf = _compute_tfidf([query_text])[0]

        scored: List[tuple[float, HindsightEntry]] = []

        for entry in self._entries:
            # TF-IDF cosine similarity
            tfidf_sim = _cosine_similarity(query_tfidf, entry.keyword_tfidf)

            # Keyword Jaccard overlap
            keyword_sim = _keyword_overlap_score(task_keywords, entry.retrieval_keywords)

            # Rubric similarity
            rubric_sim = self.get_rubric_similarity_entry(task, entry)

            # Combined score (weighted average)
            combined = 0.5 * tfidf_sim + 0.3 * keyword_sim + 0.2 * rubric_sim

            if combined >= similarity_threshold and rubric_sim >= rubric_similarity_threshold:
                scored.append((combined, entry))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def sample(
        self,
        batch_size: int = 4,
        similarity_threshold: float = 0.5,
        task: Optional[str] = None,
    ) -> List[HindsightEntry]:
        """
        Sample a batch of hindsight entries for BC-style policy update.

        If ``task`` is provided, prioritises entries similar to the task
        (via ``retrieve``). Otherwise, samples uniformly at random from
        the store.

        Parameters
        ----------
        batch_size : int
            Number of entries to sample. Defaults to 4.
        similarity_threshold : float
            Minimum similarity to include an entry when task is provided.
            Defaults to 0.5.
        task : str, optional
            If provided, retrieve similar entries rather than random sampling.

        Returns
        -------
        List[HindsightEntry]
            Sampled entries (may be fewer than batch_size if store is small
            or not enough entries meet the threshold).
        """
        if not self._entries:
            return []

        if task:
            candidates = self.retrieve(
                task, top_k=batch_size * 2, similarity_threshold=similarity_threshold
            )
            if not candidates:
                # Fall back to random if no similar entries found
                candidates = list(self._entries)
            # Shuffle and take batch_size
            import random
            random.shuffle(candidates)
            return candidates[:batch_size]
        else:
            import random
            return random.sample(self._entries, min(batch_size, len(self._entries)))

    def get_rubric_similarity(
        self,
        entry: HindsightEntry,
        new_milestone: Milestone,
    ) -> float:
        """
        Compute rubric-level similarity between a stored entry and a new milestone.

        Compares the failed milestone's success_rubric with the new milestone's
        success_rubric using keyword overlap.

        Parameters
        ----------
        entry : HindsightEntry
            The stored hindsight entry.
        new_milestone : Milestone
            The new milestone to compare against.

        Returns
        -------
        float
            Similarity score in [0.0, 1.0].
        """
        old_keywords = _extract_keywords(entry.failed_milestone.success_rubric)
        new_keywords = _extract_keywords(new_milestone.success_rubric)
        return _keyword_overlap_score(old_keywords, new_keywords)

    def get_rubric_similarity_entry(
        self,
        task: str,
        entry: HindsightEntry,
    ) -> float:
        """
        Compute rubric-level similarity between a task string and a stored entry.

        Compares the task with the entry's failed milestone description using
        keyword overlap.

        Parameters
        ----------
        task : str
            The new task description.
        entry : HindsightEntry
            The stored hindsight entry.

        Returns
        -------
        float
            Similarity score in [0.0, 1.0].
        """
        task_keywords = _extract_keywords(task)
        milestone_keywords = _extract_keywords(entry.failed_milestone.description)
        return _keyword_overlap_score(task_keywords, milestone_keywords)

    def export(self, path: Optional[Path] = None) -> None:
        """
        Export the entire store to a JSON file.

        Parameters
        ----------
        path : pathlib.Path, optional
            Path to export to. Defaults to ``{store_dir}/store.json``.
        """
        out_path = Path(path) if path else self._store_dir / "store.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: Optional[Path] = None) -> int:
        """
        Load entries from a JSON file into the store.

        Parameters
        ----------
        path : pathlib.Path, optional
            Path to load from. Defaults to ``{store_dir}/store.json``.

        Returns
        -------
        int
            Number of entries loaded.
        """
        in_path = Path(path) if path else self._store_dir / "store.json"

        if not in_path.exists():
            return 0

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = [HindsightEntry.from_dict(d) for d in data.get("entries", [])]
        self._entries.extend(entries)

        # Recompute TF-IDF for all entries
        if self._entries:
            all_texts = [
                e.original_task + " " + e.failed_milestone.description
                for e in self._entries
            ]
            tfidf_vecs = _compute_tfidf(all_texts)
            for i, entry in enumerate(self._entries):
                entry.keyword_tfidf = tfidf_vecs[i]

        return len(entries)

    # ------------------------------------------------------------------
    # Statistics / introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the store."""
        verdict_counts: Dict[str, int] = {}
        for entry in self._entries:
            key = entry.critic_result.verdict.value
            verdict_counts[key] = verdict_counts.get(key, 0) + 1

        return {
            "total_entries": len(self._entries),
            "unique_episodes": len(set(e.episode_id for e in self._entries)),
            "verdict_distribution": verdict_counts,
            "store_dir": str(self._store_dir),
            "max_entries": self._max_entries,
        }

    def clear(self) -> int:
        """
        Clear all entries from the store.

        Returns
        -------
        int
            Number of entries that were cleared.
        """
        count = len(self._entries)
        self._entries.clear()
        return count

    def __len__(self) -> int:
        """Return the number of entries in the store."""
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"HindsightStore(entries={len(self._entries)}, "
            f"max_entries={self._max_entries}, dir={self._store_dir})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hindsight_label(
        milestone: Milestone,
        result: MilestoneCriticResult,
    ) -> str:
        """
        Generate a HeRL-style hindsight relabeling string.

        The format is: "What if [{milestone description}] was the actual goal?"
        with the verdict appended as context.
        """
        base = f"What if [{milestone.description}] was the actual goal?"
        verdict_note = f" (verdict: {result.verdict.value}, confidence={result.confidence:.2f})"
        return base + verdict_note
