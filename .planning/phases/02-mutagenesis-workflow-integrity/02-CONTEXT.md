# Phase 2: Mutagenesis Workflow Integrity - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix three correctness gaps in the mutagenesis workflow: enforce the Phase 1→2 approval gate so no mutations execute without explicit user sign-off, persist Phase 1 results to disk so state survives browser close, and verify that expert panel backend overrides actually reach all agents. No new agent types. No new pipeline steps. This phase is about making existing functionality correct and reliable.

</domain>

<decisions>
## Implementation Decisions

### Approval gate UX
- "Approve & Continue" button lives at the bottom of the suggestions table (user scrolls through all suggestions first)
- If clicked with no mutations selected: show inline error message ("Select at least one mutation to approve"), block — do not proceed
- If user tries to start Phase 2 without going through the approval step: show a confirmation dialog ("No mutations approved — are you sure?"), not a hard block; user can override if they want to use fallback path

### Disk persistence format
- Format: JSON (human-readable, easy to inspect/debug, no extra dependencies)
- Save timing: auto-save when Phase 1 pipeline completes (not on approve click — the raw suggestions + low-confidence positions are saved immediately after Phase 1 agents finish)
- Load behavior: auto-load silently when user returns to a job with Phase 1 results on disk; show a subtle indicator "Loaded from previous session" but no prompt

### Backend override verification
- Observable evidence: BOTH the per-call timing log AND a UI preview before Phase 2 starts
  - Timing log should include the model name: `[LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s (qwen2.5:14b)`
  - UI shows "Using: [model] @ [provider]" confirmation before Phase 2 button is active
- Override scope: ALL Phase 2 mutagenesis agents use the selected backend (not just review agents)

### Claude's Discretion
- Exact JSON schema for persisted Phase 1 state (what fields, how nested)
- File naming convention for the persisted state file within the job directory
- Exact wording of the confirmation dialog for the no-approval case
- How the timing log model name is appended (prefix, suffix, or bracketed field)

</decisions>

<specifics>
## Specific Ideas

- No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-mutagenesis-workflow-integrity*
*Context gathered: 2026-02-21*
