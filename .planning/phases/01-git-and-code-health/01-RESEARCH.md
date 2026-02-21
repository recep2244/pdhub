# Phase 1: Git & Code Health - Research

**Researched:** 2026-02-21
**Domain:** Git hygiene, Python module tracking, import alias patterns
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Audit ALL import paths before committing mutagenesis_agents.py: orchestrator, registry, CLI, and web pages
- After committing, run the existing pipeline integrity tests (10-step and 12-step flows) to confirm nothing broke
- Diff `.bak` against current `10_mutation_scanner.py` before deleting — confirm nothing was accidentally lost in the refactor
- Add `*.bak` to `.gitignore` after deletion to prevent future occurrences
- Rename `MutagenesiReportAgent` → `MutagenesisPipelineReportAgent` as the canonical class name
- Add import alias in same file: `MutagenesiReportAgent = MutagenesisPipelineReportAgent` (no deprecation warning — silent alias is sufficient)
- Update ALL call sites: orchestrator, registry, tests, web pages — use canonical name everywhere
- Add an import test confirming both old and new names are importable after the rename

### Claude's Discretion
- Commit message wording and granularity (one commit per fix vs combined)
- Exact diff tool/approach for comparing .bak vs current

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GIT-01 | `mutagenesis_agents.py` committed to git with orchestrator/registry imports verified | Import audit complete: only orchestrator.py (lines 145-150) and registry.py (lines 87-96) import from this module. No CLI or web page imports it directly. All 17 tests currently pass. |
| GIT-02 | `10_mutation_scanner.py.bak` deleted and confirmed safe to remove | Diff analysis complete: .bak is 124KB from Feb 10 15:56; current is 128KB from Feb 10 15:58. Diff of ~1150 changed lines confirms current is a superset — refactored predictor selection into `_render_manual_tab_settings()`, added example sequences and change-sequence button. Nothing exists only in .bak. Safe to delete. |
| GIT-03 | `MutagenesiReportAgent` class renamed to `MutagenesisPipelineReportAgent` with backward-compatible import alias | All call sites identified: definition in mutagenesis_agents.py line 387, import+use in orchestrator.py lines 148+160, import+register in registry.py lines 92+96. No web pages or CLI files reference this class by name. Tests use only the string key "mutagenesis_report", not the class name. |
</phase_requirements>

---

## Summary

This phase contains three mechanical fixes with zero feature development. The work is well-understood and low-risk because all three hazards are isolated: the untracked file, the orphaned backup, and the typo class name each have a narrow blast radius.

The import audit (GIT-01) reveals that `mutagenesis_agents.py` is imported only in two files: `orchestrator.py` (inside `_build_mutagenesis_post_approval_pipeline`, a lazy import at lines 145-150) and `registry.py` (inside `_register_mutagenesis_agents`, also lazy at lines 87-96). No web pages or CLI commands import it directly. All 17 pipeline integrity tests currently pass with the module untracked — meaning Python resolves it from the working tree. Git-committing the file changes nothing at runtime; it only adds it to version history.

The backup file (GIT-02) was created Feb 10 15:56, two minutes before the current version (Feb 10 15:58). The diff confirms the current file is the more advanced version: it introduces `_render_manual_tab_settings()` as a proper function, adds example sequence quick-load buttons, and adds a "Change Sequence" action. No content is exclusively in the `.bak`. The `.gitignore` currently has `env.bak/` and `venv.bak/` but NOT `*.bak` — this must be added.

The rename (GIT-03) touches exactly 3 source files. Tests reference this agent only by its string name `"mutagenesis_report"` (never by class name), so no test changes are needed beyond adding the new import-alias test. The silent alias pattern is the correct Python idiom for this situation.

**Primary recommendation:** Execute three separate, focused git commits — one per requirement. This makes each fix atomic and independently revertable.

---

## Standard Stack

This phase uses only git and Python standard library patterns. No new dependencies.

### Core
| Tool/Pattern | Version | Purpose | Why Standard |
|--------------|---------|---------|--------------|
| git add / commit | system git | Track untracked file | Standard VCS operation |
| git rm | system git | Remove .bak file from working tree | Preferred over manual `rm` + `git add` — single command removes and stages |
| Python import alias | Python 3.x | Backward-compatible class rename | `OldName = NewName` is the canonical Python idiom for this pattern |
| pytest | already in project | Verify no regressions | All 17 tests currently pass; run again after each change |

### No Installation Required
All tooling is already present. No `pip install` needed for this phase.

---

## Architecture Patterns

### Pattern 1: Lazy Import Scope (how mutagenesis_agents.py is used)

**What:** Both `orchestrator.py` and `registry.py` import from `mutagenesis_agents` inside function bodies, not at module top level. This means import errors are deferred until the function is called.

**Impact on GIT-01:** Committing the file will not cause any import-time errors during test collection or module import. The lazy pattern means Python will only load `mutagenesis_agents.py` when `_build_mutagenesis_post_approval_pipeline()` or `_register_mutagenesis_agents()` is called. Tests that exercise `mode="mutagenesis_post"` or `AgentRegistry.list_names()` will trigger the import.

```python
# orchestrator.py — lines 145-150 (lazy import inside function)
def _build_mutagenesis_post_approval_pipeline(...):
    from protein_design_hub.agents.mutagenesis_agents import (
        MutationExecutionAgent,
        MutationComparisonAgent,
        MutagenesiReportAgent,  # this becomes MutagenesisPipelineReportAgent after GIT-03
    )
```

```python
# registry.py — lines 87-96 (lazy import inside function)
def _register_mutagenesis_agents() -> None:
    from protein_design_hub.agents.mutagenesis_agents import (
        MutationExecutionAgent,
        MutationComparisonAgent,
        MutagenesiReportAgent,  # update to canonical name
    )
```

### Pattern 2: Silent Import Alias (GIT-03 rename pattern)

**What:** In Python, the canonical pattern for renaming a class while preserving backward compatibility is a module-level alias in the same file where the class is defined.

**When to use:** Any rename where external code may reference the old name by string (e.g., serialized configs, documentation, other imports not yet updated).

```python
# In mutagenesis_agents.py — after the class definition

class MutagenesisPipelineReportAgent(BaseAgent):
    """Generate comprehensive mutagenesis report. ..."""
    name = "mutagenesis_report"
    # ... (unchanged implementation)


# Backward-compatible alias — silent, no DeprecationWarning
MutagenesiReportAgent = MutagenesisPipelineReportAgent
```

**Why silent alias:** The CONTEXT.md specifies "no deprecation warning — silent alias is sufficient." The class was never part of a public API (it's an internal agent), and the typo was caught before any external consumers stabilized against it.

### Pattern 3: Import Test for Both Names

**What:** A simple importability test confirms both the canonical name and alias are accessible after the rename.

```python
# In tests/test_agent_pipeline_integrity.py — new test to add

def test_mutagenesis_report_agent_importable_by_both_names():
    """Both canonical and legacy name must be importable after the rename."""
    from protein_design_hub.agents.mutagenesis_agents import (
        MutagenesisPipelineReportAgent,
        MutagenesiReportAgent,
    )
    assert MutagenesiReportAgent is MutagenesisPipelineReportAgent
```

### Pattern 4: .gitignore Entry for .bak Files

**What:** Add `*.bak` to project `.gitignore` to prevent future backup files from polluting the tracked/untracked state.

**Current state:** The `.gitignore` contains `env.bak/` and `venv.bak/` (directory-level) but NOT `*.bak` (file-level). The `.bak` file currently shows as `??` (untracked) in git status, which is noise that distracts from real untracked files.

```
# Add to .gitignore — Temporary files section
*.bak
```

### Anti-Patterns to Avoid
- **Renaming with DeprecationWarning:** The CONTEXT.md explicitly specifies a silent alias. Adding a `warnings.warn()` call would create unnecessary noise in test output and is out of scope.
- **Committing all three fixes in one commit:** Combining commits makes individual rollback impossible. Three atomic commits keep git history clean and each requirement independently traceable.
- **Using `git mv` for the .bak deletion:** `git mv` is for renaming tracked files. The `.bak` is untracked — just `rm` it and add `*.bak` to `.gitignore`.
- **Updating call sites before the alias exists:** Always define the alias in `mutagenesis_agents.py` first, then update call sites. Reversing the order creates a window where the canonical name doesn't exist.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Verify .bak is safe to delete | Custom script comparing file hashes | `diff old new` | diff shows exact semantic differences; hash comparison only shows binary equivalence |
| Verify import works after rename | Complex import machinery test | Simple `from module import Name; assert Name is OtherName` | Python's import system handles everything; the test just proves the alias works |
| Prevent future .bak files | Git pre-commit hook | `.gitignore` entry | Hooks are heavy; gitignore is the standard mechanism for file type exclusions |

**Key insight:** All three tasks are mechanical git/Python operations. No custom tooling is needed.

---

## Common Pitfalls

### Pitfall 1: Import Order — Rename Before Updating Call Sites
**What goes wrong:** If you update `orchestrator.py` and `registry.py` to import `MutagenesisPipelineReportAgent` before adding the alias in `mutagenesis_agents.py`, the import fails with `ImportError`.
**Why it happens:** Python evaluates imports at call time (lazy pattern here), but the name must exist in the module namespace when the import statement executes.
**How to avoid:** Sequence is: (1) Rename class in `mutagenesis_agents.py` and add alias in same file, (2) then update `orchestrator.py` and `registry.py` imports, (3) then run tests.
**Warning signs:** `ImportError: cannot import name 'MutagenesisPipelineReportAgent'` from `mutagenesis_agents`.

### Pitfall 2: Forgetting the Alias Breaks the Old Name
**What goes wrong:** After renaming the class, forgetting to add `MutagenesiReportAgent = MutagenesisPipelineReportAgent` breaks `registry.py`'s import (which still uses the old name until updated), and any external code that might reference the old name.
**Why it happens:** The class definition has a new name; the old symbol no longer exists in the module namespace.
**How to avoid:** Immediately after the class definition in `mutagenesis_agents.py`, add the alias line before touching any other file.
**Warning signs:** `ImportError: cannot import name 'MutagenesiReportAgent'` during `AgentRegistry.list_names()` or `mode="mutagenesis_post"` pipeline creation.

### Pitfall 3: .bak Delete Without .gitignore Update
**What goes wrong:** Deleting `10_mutation_scanner.py.bak` makes it disappear from `git status ??` listing, but next time someone creates a `.bak` file in the project, it shows up as untracked again.
**Why it happens:** `git status` shows untracked files that aren't gitignored.
**How to avoid:** After deleting the file, add `*.bak` to `.gitignore` in the same commit.
**Warning signs:** `.bak` files appearing in `git status` output in future sessions.

### Pitfall 4: Tests Fail If mutagenesis_agents.py Has Syntax Errors
**What goes wrong:** If `mutagenesis_agents.py` has any Python syntax error (e.g., from an edit gone wrong), the `test_mutagenesis_post_pipeline_has_4_steps` and `test_registry_exposes_mutagenesis_agents` tests will fail with `ImportError`, not a helpful `AssertionError`.
**Why it happens:** Lazy imports fail at the point of first call, which in tests is inside the constructor.
**How to avoid:** After any edit to `mutagenesis_agents.py`, run `python -c "import protein_design_hub.agents.mutagenesis_agents"` as a quick syntax check before running the full test suite.
**Warning signs:** `ImportError` or `SyntaxError` in test output instead of assertion failures.

### Pitfall 5: Current Tests Pass Without mutagenesis_agents.py Being Tracked
**What goes wrong:** Assuming that because all 17 tests pass now, the module is "fine" and doesn't need to be committed. This is false — on a fresh clone (or after `git clean -fdx`), the file won't exist and the pipeline will break.
**Why it happens:** pytest runs against the working tree, not against what's committed. Untracked files are visible to Python but not to git.
**How to avoid:** Commit the file (GIT-01 is the fix). Verify with `git status` that `mutagenesis_agents.py` moves from `??` to clean.
**Warning signs:** `?? src/protein_design_hub/agents/mutagenesis_agents.py` in `git status`.

---

## Code Examples

### GIT-01: Commit the untracked file

```bash
# Verify import paths first (no output = no import errors found in wrong places)
grep -rn "from.*mutagenesis_agents" src/ tests/
# Expected output: only orchestrator.py and registry.py

# Stage and commit
git add src/protein_design_hub/agents/mutagenesis_agents.py
git commit -m "chore: track mutagenesis_agents.py in version control

Adds the untracked computational mutagenesis agents module to git.
Verified imports: only orchestrator.py and registry.py reference this
module via lazy imports. All 17 pipeline integrity tests pass."
```

### GIT-02: Delete .bak and update .gitignore

```bash
# Confirm .bak is safe (diff should show current has MORE content, not less)
diff src/protein_design_hub/web/pages/10_mutation_scanner.py.bak \
     src/protein_design_hub/web/pages/10_mutation_scanner.py | head -50

# Delete the backup
rm src/protein_design_hub/web/pages/10_mutation_scanner.py.bak

# Add *.bak to .gitignore
# (append to the "Temporary files" section)
echo "*.bak" >> .gitignore

# Stage and commit
git add .gitignore
git rm --cached src/protein_design_hub/web/pages/10_mutation_scanner.py.bak 2>/dev/null || true
git commit -m "chore: delete orphaned .bak backup and ignore *.bak files

10_mutation_scanner.py.bak (122KB, Feb 10 15:56) confirmed safe to
remove — current file (128KB, Feb 10 15:58) is a superset containing
all .bak content plus the _render_manual_tab_settings() refactor and
example sequence loader additions. Added *.bak to .gitignore."
```

### GIT-03: Rename class with backward-compatible alias

Step 1 — Update `mutagenesis_agents.py`:

```python
# Change line 387 from:
class MutagenesiReportAgent(BaseAgent):

# To:
class MutagenesisPipelineReportAgent(BaseAgent):

# Then add alias immediately after the class ends (after line 559):
# Backward-compatible alias — MutagenesiReportAgent was a typo
MutagenesiReportAgent = MutagenesisPipelineReportAgent
```

Step 2 — Update `orchestrator.py` (lines 145-150):

```python
from protein_design_hub.agents.mutagenesis_agents import (
    MutationExecutionAgent,
    MutationComparisonAgent,
    MutagenesisPipelineReportAgent,  # was: MutagenesiReportAgent
)
# ...
MutagenesisPipelineReportAgent(progress_callback=progress_callback),  # line 160
```

Step 3 — Update `registry.py` (lines 89-96):

```python
from protein_design_hub.agents.mutagenesis_agents import (
    MutationExecutionAgent,
    MutationComparisonAgent,
    MutagenesisPipelineReportAgent,  # was: MutagenesiReportAgent
)
AgentRegistry.register("mutation_execution", MutationExecutionAgent)
AgentRegistry.register("mutation_comparison", MutationComparisonAgent)
AgentRegistry.register("mutagenesis_report", MutagenesisPipelineReportAgent)  # was: MutagenesiReportAgent
```

Step 4 — Add import test to `tests/test_agent_pipeline_integrity.py`:

```python
def test_mutagenesis_report_agent_importable_by_both_names():
    """Canonical class and backward-compat alias must both be importable."""
    from protein_design_hub.agents.mutagenesis_agents import (
        MutagenesisPipelineReportAgent,
        MutagenesiReportAgent,
    )
    assert MutagenesiReportAgent is MutagenesisPipelineReportAgent
```

---

## State of the Art

| Old State | Current/Fixed State | Fix | Impact |
|-----------|---------------------|-----|--------|
| `mutagenesis_agents.py` untracked (`??`) | Tracked, in git history | `git add` + commit | Fresh clone no longer breaks mutagenesis pipeline |
| `10_mutation_scanner.py.bak` in working tree as untracked | Deleted; `*.bak` gitignored | `rm` + `.gitignore` entry | `git status` stays clean; no future backup noise |
| `MutagenesiReportAgent` (typo class name) | `MutagenesisPipelineReportAgent` canonical; old name as alias | Class rename in mutagenesis_agents.py + call site updates | Consistent naming; both names import correctly |

---

## Open Questions

None. The research is complete and all questions are answered by direct code inspection.

1. **Does the .bak contain anything not in the current file?**
   - What we know: Diff shows 1150 changed lines; the current file is 4KB larger than the .bak; tail of diff shows current has `_render_manual_tab_settings()`, `_EXAMPLE_SEQUENCES`, "Change Sequence" button — all additions.
   - Conclusion: Current file is strictly a superset. `.bak` is safe to delete.

2. **Are there any web page or CLI files that import MutagenesiReportAgent by name?**
   - What we know: Grep of all `src/` and `tests/` for both class names finds: mutagenesis_agents.py (line 387, definition), orchestrator.py (lines 148, 160), registry.py (lines 92, 96). No other files.
   - Conclusion: Only 3 source files need updates for GIT-03.

3. **Do any tests reference MutagenesiReportAgent by class name?**
   - What we know: `test_agent_pipeline_integrity.py` checks pipeline step names as strings (`"mutagenesis_report"`), never imports or references the class by name.
   - Conclusion: Only the new import-alias test needs to reference the class names.

---

## Execution Order (CRITICAL)

The three requirements must be done in this order:

```
GIT-01 → GIT-02 → GIT-03
```

**Rationale:**
- GIT-01 first: Once committed, `mutagenesis_agents.py` is safely in git history. The rename (GIT-03) will then be part of the same file's history.
- GIT-02 is independent of GIT-03 but doing it second keeps the commits logically grouped.
- GIT-03 last: The class rename is the only change that touches multiple files atomically. All tests should pass before starting GIT-03.

**Within GIT-03, the file edit order is:**
1. `mutagenesis_agents.py` — rename class, add alias (both names now exist)
2. `orchestrator.py` — update import to canonical name
3. `registry.py` — update import to canonical name
4. `tests/test_agent_pipeline_integrity.py` — add alias import test
5. Run `pytest tests/test_agent_pipeline_integrity.py` — verify all 18 tests pass

---

## Sources

### Primary (HIGH confidence)
- Direct file inspection: `src/protein_design_hub/agents/mutagenesis_agents.py` (559 lines, class at line 387)
- Direct file inspection: `src/protein_design_hub/agents/orchestrator.py` (import at lines 145-150, instantiation at line 160)
- Direct file inspection: `src/protein_design_hub/agents/registry.py` (import at lines 89-93, register at line 96)
- Direct file inspection: `tests/test_agent_pipeline_integrity.py` (17 tests, all passing)
- `git status` output confirming `?? mutagenesis_agents.py` and `?? 10_mutation_scanner.py.bak`
- `diff` of `.bak` vs current — 1150 changed lines, current is superset
- `pytest` run confirming 17/17 tests pass with current working tree state

### Secondary (MEDIUM confidence)
- `.gitignore` file inspection confirming `*.bak` is not currently excluded
- Python language documentation: `OldName = NewName` is the standard module-level alias pattern

### Tertiary (LOW confidence)
- None — all findings are from direct code inspection, no WebSearch required for this phase

---

## Metadata

**Confidence breakdown:**
- GIT-01 (commit untracked file): HIGH — direct `git status` and grep confirm exact state
- GIT-02 (delete .bak): HIGH — diff confirms current is superset; .gitignore confirmed to lack `*.bak`
- GIT-03 (rename with alias): HIGH — all 4 reference locations identified by grep; Python alias pattern is well-established

**Research date:** 2026-02-21
**Valid until:** This research reflects a specific working-tree state. It remains valid until any of the 3 target files are modified outside of this phase's tasks.
