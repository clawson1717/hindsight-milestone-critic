# HiMCA Implementation Plan

## Step 1: Project Scaffolding ✅ [DONE]
- GitHub repo created
- README.md with architecture
- src/ package structure
- requirements.txt

## Step 2: Milestone Decomposer ✅ [DONE — PR #1, 2026-03-27]
- `src/decompose.py`: `MilestoneDecomposer` → `List[Milestone]`
- Task → list of verifiable subgoals
- Unit tests

## Step 3: Evidence Collector ✅ [DONE — PR #2, 2026-03-30]
- `src/evidence.py`: `EvidenceCollector`
- Extract supporting evidence for each milestone from trajectory
- Structured evidence format

## Step 4: Milestone Critic ✅ [DONE — PR #3, 2026-03-30]
- `src/critic.py`: `MilestoneCritic` (OS-Themis style)
- Verdict per milestone: pass/fail/uncertain
- Boundary enforcement layer (Box Maze)

## Step 5: Hindsight Experience Store ✅ [DONE — PR #4, 2026-03-31]
- `src/hindsight.py`: `HindsightStore` (HeRL-style)
- Failed milestone → stored with rubric context
- Efficient retrieval for new tasks

## Step 6: Hindsight-Aware Policy ✅ [DONE — PR #5]
- `src/policy.py`: `HindsightAwarePolicy`
- Policy shaped away from previously-failed modes

## Step 7: Integration Loop ⬜
- `src/loop.py`: `HiMCALoop`
- Orchestrates all 6 modules
- Early termination on all-milestones-pass

## Step 8: CLI Interface ⬜
- `src/cli.py`: `python -m src.cli`
- Commands: run, critic, hindsight, benchmark

## Step 9: Integration Tests ⬜
- `tests/test_integration.py`
- End-to-end scenarios

## Step 10: Documentation + Final PR ⬜
- Full README with examples
- All steps marked [DONE]
