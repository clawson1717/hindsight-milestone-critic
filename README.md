# Hindsight Milestone Critic Agent (HiMCA)

**GitHub:** https://github.com/clawson1717/hindsight-milestone-critic

Self-correcting agent framework that combines **rigorous milestone-level verification** with **hindsight experience replay** — so agents learn from every failure.

---

## The Problem

Existing agent frameworks that use milestones (OS-Themis, MiRA) discard failed milestones after evaluation. But **failed milestones are the most valuable learning signal** — they tell you exactly what went wrong, why, and how to avoid it next time.

**HiMCA** solves this by treating every failed milestone as a hindsight experience entry, shaping the policy away from previously-failed modes.

---

## Architecture

```
Task Input
    │
    ▼
┌─────────────────────────┐
│  Milestone Decomposer   │ → List[Milestone]
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Evidence Collector     │ → Evidence chains per milestone
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Milestone Critic      │ ← Boundary Enforcement (Box Maze)
│   (OS-Themis style)     │ → VERDICT per milestone
└─────────────────────────┘
    │
    ├─ PASS → continue
    │
    └─ FAIL → HindsightStore.add() ← stored with rubric context
                                     (HeRL-style)
    │
    ▼
┌─────────────────────────┐
│ Hindsight-Aware Policy  │ ← Context enriched with relevant
│   (Policy shaper)       │   hindsight entries
└─────────────────────────┘
    │
    ▼
  Next Attempt
```

---

## Key Innovation

| Feature | OS-Themis | HeRL | Box Maze | **HiMCA** |
|---------|-----------|------|----------|-----------|
| Milestone decomposition | ✓ | ✗ | ✗ | **✓** |
| Evidence-chain critic | ✓ | ✗ | ✗ | **✓** |
| Hindsight from failures | ✗ | ✓ | ✗ | **✓** |
| Boundary enforcement | ✗ | ✗ | ✓ | **✓** |

**Novel combination:** OS-Themis-style milestone critics + HeRL hindsight experience + Box Maze boundary enforcement. No existing framework has this combination.

---

## Implementation Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project scaffolding | ✅ [DONE] |
| 2 | Milestone Decomposer | ✅ [DONE] |
| 3 | Evidence Collector | ✅ [DONE] |
| 4 | Milestone Critic (OS-Themis) | ✅ [DONE] |
| 5 | Hindsight Experience Store | ⬜ |
| 6 | Hindsight-Aware Policy | ⬜ |
| 7 | Integration Loop | ⬜ |
| 8 | CLI Interface | ⬜ |
| 9 | Integration Tests | ⬜ |
| 10 | Documentation + Final PR | ⬜ |

---

## Quick Start

```bash
# Clone
git clone https://github.com/clawson1717/hindsight-milestone-critic.git
cd hindsight-milestone-critic

# Install
pip install -e .

# Run
python -m src.cli run "Your complex multi-step task here"

# Critic mode — evaluate a trajectory
python -m src.cli critic --trajectory path/to/trajectory.json

# View hindsight store
python -m src.cli hindsight --list
```

---

## CLI Commands

- `run <task>` — Execute task through HiMCA loop
- `critic --trajectory <path>` — Evaluate a trajectory's milestones
- `hindsight --list` — List all stored hindsight entries
- `benchmark` — Run HiMCA on benchmark suite

---

## File Structure

```
src/
  __init__.py
  decompose.py    # Milestone decomposition
  evidence.py     # Evidence collection
  critic.py       # Milestone critic (OS-Themis style)
  hindsight.py    # Hindsight experience store
  policy.py       # Hindsight-aware policy
  loop.py         # Main HiMCA loop
  cli.py          # CLI interface
tests/
  conftest.py
  test_decompose.py
  test_evidence.py
  test_critic.py
  test_hindsight.py
  test_integration.py
```

---

## Prior Art

- **OS-Themis** (arXiv:2603.19191) — Multi-agent critic with milestone decomposition for GUI agents
- **HeRL** (2026-03-23 digest) — Hindsight experience from failed trajectories for LLM RL
- **Box Maze** (arXiv:2603.19182) — Boundary enforcement layers for reliable LLM reasoning
