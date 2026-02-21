# Codebase Concerns

**Analysis Date:** 2026-02-21

## Tech Debt

**Untracked mutagenesis_agents.py module:**
- Issue: `src/protein_design_hub/agents/mutagenesis_agents.py` (559 lines) is untracked in git but imported by orchestrator and registry. Module is not yet committed to version control.
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py`, `src/protein_design_hub/agents/orchestrator.py` (lines 145-150), `src/protein_design_hub/agents/registry.py`
- Impact: Missing from git history; if lost locally, pipeline breaks. Integration with Phase 2 (mutagenesis_post) becomes fragile.
- Fix approach: Commit the module and audit all imports to ensure registry and orchestrator properly expose the agents.

**Large mutation_scanner page (2844 lines):**
- Issue: `src/protein_design_hub/web/pages/10_mutation_scanner.py` is 2.8K lines — monolithic UI page combining expert backend override UI, phase detection, approval step logic, saturation mutagenesis, and multi-mutation design all in one file.
- Files: `src/protein_design_hub/web/pages/10_mutation_scanner.py`
- Impact: Hard to maintain, difficult to test individual features, navigation through 2844 lines is cumbersome. Changes to approval step require searching through full file.
- Fix approach: Split into smaller modules: `_mutation_scanner_core.py` (saturation mutagenesis), `_mutation_approval.py` (Phase 1→2 transition), `_expert_backend_ui.py` (LLM expert panel override).

**Incomplete refactor with backup file:**
- Issue: `src/protein_design_hub/web/pages/10_mutation_scanner.py.bak` (122K, 2,768 lines) exists as backup from 2026-02-10 15:56. Recent changes moved predictor selection and baseline comparison into `_render_manual_tab_settings()` function but backup suggests incomplete migration.
- Files: `src/protein_design_hub/web/pages/10_mutation_scanner.py` (current), `src/protein_design_hub/web/pages/10_mutation_scanner.py.bak` (backup)
- Impact: Unclear what was refactored; backup may mask future merge conflicts or accidental reversions. Maintenance burden.
- Fix approach: Delete backup file. Verify all manual-tab settings are correctly extracted into the new function. Add git hooks to reject `.bak` files.

## Known Bugs

**Missing approval gate between Phase 1 and Phase 2:**
- Issue: The UI allows jumping from `mutagenesis_pre` (Phase 1, suggests mutations) to `mutagenesis_post` (Phase 2, executes them) without enforcing user approval of suggestions. The `_render_approval_step()` function in `src/protein_design_hub/web/pages/10_mutation_scanner.py` (line 1609) displays suggestions but does not validate that user has explicitly reviewed and approved before Phase 2 is triggered.
- Files: `src/protein_design_hub/web/pages/10_mutation_scanner.py` (lines 1609-1650, 1735), `src/protein_design_hub/agents/mutagenesis_agents.py` (MutationExecutionAgent expects `context.extra["approved_mutations"]`)
- Trigger: User finishes Phase 1, suggestions appear in UI, user clicks "Run Phase 2" without explicitly approving mutations.
- Workaround: UI shows warning "No approved mutations and no low-confidence positions available" only if both are missing; silent fallback to saturation at low-confidence positions occurs.
- Impact: Users may execute mutations based on stale/unreviewed suggestions; audit trail is incomplete.
- Fix approach: Add explicit "Approve & Continue" button. Block Phase 2 execution if `context.extra["approved_mutations"]` is empty and `context.extra["baseline_low_confidence_positions"]` is not set.

**Per-expert backend override mechanism lacks clarity:**
- Issue: The expert backend override UI in `src/protein_design_hub/web/pages/10_mutation_scanner.py` (lines 850–982) allows users to select different LLM providers/models for mutation review panels (qwen, deepseek, custom, etc.) but the mechanism for passing these overrides to Phase 2 agents is not documented. The session state keys (`mut_review_provider`, `mut_review_model`, `mut_review_custom_provider`) are stored but it's unclear how they flow into `LLMMutationSuggestionAgent`, `LLMMutationResultsAgent`, etc.
- Files: `src/protein_design_hub/web/pages/10_mutation_scanner.py` (lines 850–982), `src/protein_design_hub/agents/llm_guided.py` (LLMBaselineReviewAgent, LLMMutationSuggestionAgent, LLMMutationResultsAgent), `src/protein_design_hub/web/agent_helpers.py`
- Impact: Overrides may be silently ignored; users cannot confidently select specific backends for mutagenesis expert panels; inconsistent behavior across the pipeline.
- Fix approach: Add explicit parameter passing from session state to orchestrator kwargs in Phase 2 startup. Document the override flow (session → web page → CLI kwargs → agent constructor → meeting manager).

## Security Considerations

**LLM provider credentials in session state:**
- Risk: Custom LLM provider configuration (API keys, endpoint URLs) may be stored in Streamlit session state if user enters sensitive provider details in the "Custom provider ID" field. Session state can leak via memory or logs.
- Files: `src/protein_design_hub/web/pages/10_mutation_scanner.py` (lines 907–919), `src/protein_design_hub/core/config.py` (LLM provider presets)
- Current mitigation: Only provider name (e.g., "openrouter") is stored; actual API keys are expected to come from environment variables (`DEEPSEEK_API_KEY`).
- Recommendations: (1) Explicitly document that custom provider config should NOT contain API keys. (2) Add validation to reject provider strings that look like credentials (e.g., "sk-..." tokens). (3) Add warning message in UI if user enters anything that looks like a secret.

## Performance Bottlenecks

**Full saturation mutagenesis at low-confidence positions (fallback path):**
- Problem: If LLM mutation plan parsing fails, `MutationExecutionAgent.run()` in `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 84–95) falls back to saturation mutagenesis at top 5 low-confidence positions. For a 300-residue protein, 5 positions × 19 mutations = 95 predictions (+ WT baseline). Each prediction adds 30–60s overhead, totaling 50–100 minutes.
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 84–95, 158–178), `src/protein_design_hub/agents/llm_guided.py` (LLMMutationSuggestionAgent parsing)
- Cause: Parser fallback is silent; no early warning to user that "targeted mutations were unparseable, proceeding with expensive saturation".
- Improvement path: (1) Log JSON parse errors with helpful details (e.g., "missing 'positions' key"). (2) Add progress callback reporting total mutations before execution. (3) Offer UI option to limit saturation to top 2 positions instead of top 5.

**MutationScanner._build_scanner() uses lazy loading with TypeError fallback:**
- Problem: `_build_scanner()` in `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 25–44) attempts to instantiate MutationScanner with `run_openstructure_comprehensive=True`, catches `TypeError` (indicating old version), then falls back to manual `setattr()`. This pattern is fragile and hides version mismatches.
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 25–44)
- Cause: MutationScanner API may differ between installed versions; lazy fallback masks real errors.
- Improvement path: (1) Add explicit version check at module init. (2) Raise `ImportError` with helpful message if scanner version is too old. (3) Document minimum MutationScanner version required.

## Fragile Areas

**LLMBaselineReviewAgent produces per-residue pLDDT analysis that Phase 2 depends on:**
- Files: `src/protein_design_hub/agents/llm_guided.py` (LLMBaselineReviewAgent, lines 1041–1183), `src/protein_design_hub/agents/mutagenesis_agents.py` (MutationExecutionAgent fallback, lines 84–95)
- Why fragile: Phase 1 LLM team identifies low-confidence positions (pLDDT < 70) and stores them in `context.extra["baseline_low_confidence_positions"]` (line 1177). Phase 2 MutationExecutionAgent (line 85) silently uses this fallback list if no `approved_mutations` are provided. If the LLM team fails to identify any low-conf positions (e.g., due to prompt ambiguity or LLM confusion), Phase 2 has nothing to execute.
- Safe modification: (1) Add explicit verdict check — if `baseline_review` status is FAIL, block Phase 2 startup. (2) Log all per-residue analysis at DEBUG level before executing. (3) Add UI warning if fewer than 3 low-confidence positions were identified.
- Test coverage: Pipeline integration tests check step order (test_agent_pipeline_integrity.py) but do NOT verify that Phase 1 outputs are complete/valid before Phase 2 runs. Missing: `test_mutagenesis_phase1_generates_required_extra_keys()`, `test_mutagenesis_phase2_rejects_missing_baseline_data()`.

**MutagenesiReportAgent (typo: class name is `MutagenesiReportAgent` not `MutagenesisPipelineReportAgent`):**
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py` (line 387), `src/protein_design_hub/agents/orchestrator.py` (line 150)
- Why fragile: Typo in class name (`Mutagenesi` vs `Mutagenesis`) is inconsistent with naming conventions elsewhere. If refactoring tools or documentation rely on exact class names, the typo creates friction. Agent name in orchestrator is correctly set to `"mutagenesis_report"` (line 394), so runtime works, but the class definition is misleading.
- Safe modification: Rename to `MutagenesisPipelineReportAgent` and update all imports. Add deprecation alias for the old name.

## Scaling Limits

**OST (OpenStructure) metrics extraction may fail on large mutations:**
- Problem: `_extract_ost_metrics()` in `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 47–61) assumes OST outputs are available in `extra_metrics["ost_comprehensive"]["global"]`. If OST run fails or times out, the dict is empty or missing, and metrics silently become {}.
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 47–61, 172–175, 221–223)
- Current capacity: Works well for proteins up to ~500 residues with saturation at 1–2 positions. At 5+ positions, OST comprehensive comparison becomes a bottleneck (5 positions × 19 mutations = 95 structures to compare, each comparison ~60s on RTX 4080 laptop).
- Limit: Beyond ~10 positions, Phase 2 runtime exceeds 2+ hours. OpenStructure comparison is CPU-bound; GPU memory (12 GB RTX 4080) is not the constraint, but OST processing is slow.
- Scaling path: (1) Make OST optional with flag `--skip-openstructure` (default: enabled for ≤3 positions, disabled for >5). (2) Implement batch structure comparison API. (3) Consider async/parallel OST runs (current code is serial, lines 166–177).

## Dependencies at Risk

**MutationScanner API compatibility:**
- Risk: `src/protein_design_hub/agents/mutagenesis_agents.py` (MutationExecutionAgent) imports and uses `MutationScanner` with methods `.predict_single()`, `.scan_position()`, `.calculate_biophysical_metrics()`. If the external `protein_design_hub.analysis.mutation_scanner` module changes API (e.g., method renames, parameter changes), Phase 2 agents break silently.
- Impact: Currently fallback error handling in MutationExecutionAgent.run() (lines 178–187) catches exceptions and logs warnings, but large batches of mutation failures may not be obvious until report generation.
- Migration plan: (1) Pin exact version of `mutation_scanner` library in requirements. (2) Add compatibility layer in `_build_scanner()` to abstract API differences. (3) Add integration test that runs Phase 2 with a small sequence to catch API breaks early.

## Test Coverage Gaps

**No integration test for Phase 1 → Phase 2 transition:**
- What's not tested: The complete flow from mutagenesis_pre (Phase 1) finishing and writing `approved_mutations` to session state, to mutagenesis_post (Phase 2) starting and reading those mutations, is not tested. Only individual pipeline step orders are checked.
- Files: `tests/test_agent_pipeline_integrity.py` (lines 133–196 test pipeline definitions; missing cross-phase tests), `src/protein_design_hub/web/pages/10_mutation_scanner.py` (approval logic, lines 1609–1750 untested)
- Risk: Refactoring either phase could silently break the approval/transition mechanism.
- Priority: High — this is the critical user-facing workflow.

**Web UI approval step logic is not tested:**
- What's not tested: `_render_approval_step()` (line 1609), `_parse_approved_mutations()` (line 1701), and the button logic that transitions from Phase 1 → Phase 2 (line 1735). Streamlit UI components cannot be unit-tested easily, but the parsing logic could be extracted and tested.
- Files: `src/protein_design_hub/web/pages/10_mutation_scanner.py` (lines 1609–1750), `tests/test_web_smoke.py` (only smoke tests; no logic tests)
- Risk: Mutations approval parsing could silently fail (e.g., if dataframe columns are renamed), and Phase 2 would proceed with empty `approved_mutations`.
- Priority: Medium — add unit tests for `_parse_approved_mutations()` function.

**MutationExecutionAgent error handling is partially tested:**
- What's not tested: Scenarios where WT prediction fails (line 124), where some mutations fail but others succeed (lines 178–187), or where all mutations fail (line 260). Pipeline integrity test only checks step order, not error recovery.
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 82–274), `tests/test_agent_pipeline_integrity.py` (no mutation execution tests)
- Risk: Silent partial failures — if 15/20 mutations fail, the agent returns ok() because at least one succeeded (line 260).
- Priority: Medium — add unit tests for mutation execution failure modes.

**OST metrics extraction not tested:**
- What's not tested: The behavior of `_extract_ost_metrics()` (lines 47–61) when OST outputs are missing, malformed, or contain unexpected nesting.
- Files: `src/protein_design_hub/agents/mutagenesis_agents.py` (lines 47–61, 172–175, 221–223)
- Risk: Silent metric omissions in reports if OST dict structure changes.
- Priority: Low — defensive coding is in place (dict.get() with defaults).

**No test for LLMMutationSuggestionAgent JSON parsing fallback:**
- What's not tested: The fallback path when `_parse_mutation_plan_from_summary()` fails to extract `MUTATION_PLAN_JSON:` footer, and the agent silently switches to saturation mutagenesis at low-confidence positions.
- Files: `src/protein_design_hub/agents/llm_guided.py` (LLMMutationSuggestionAgent, lines 1186–1378), `tests/test_agent_pipeline_integrity.py` (lines 199–227 test parsing but not integration with Phase 1→2)
- Risk: Expensive fallback (saturation) is triggered silently; user sees 95 mutation predictions without warning why they weren't the 5 targeted mutations LLM suggested.
- Priority: Medium — add test for parsing failure recovery.

**No end-to-end integration test with mock LLM:**
- What's not tested: A complete Phase 1 → Phase 2 run using mock LLM agents and a real MutationScanner on a small sequence. The closest test is `test_web_smoke.py` which only checks that pages load, not that workflows execute.
- Files: `tests/test_web_smoke.py` (smoke tests only), `tests/test_agent_pipeline_integrity.py` (structure tests only)
- Risk: Refactoring agents or context shape could break the workflow without test coverage catching it.
- Priority: High — add e2e test `test_mutagenesis_phase1_to_phase2_flow_with_mock_llm()`.

## Missing Critical Features

**Phase 1→2 approval mechanism has no persistence:**
- Problem: If user approves mutations in Phase 1 and closes the browser, the `approved_mutations` stored in `context.extra` are lost. When Phase 2 is re-run later, there is no way to re-load the Phase 1 approval state. The UI may re-run Phase 1 from scratch instead.
- Blocks: Users cannot pause mutagenesis workflow across browser sessions or restart Phase 2 with previously-approved mutations.
- Fix approach: (1) Persist `approved_mutations` and `mutation_suggestions` to disk (in job directory) when Phase 1 completes. (2) Add UI option "Load previous Phase 1 results" that deserializes the approval state.

**No visibility into per-mutant OST scoring decisions:**
- Problem: OpenStructure metrics (lDDT, RMSD, QS score) are computed per mutant (line 221) and stored in results, but the UI does not clearly visualize how OST metrics informed ranking. MutationComparisonAgent's improvement_score (line 301) is 0.6×delta_plddt + 0.4×delta_local_plddt, with no OST weighting.
- Blocks: Users cannot audit why certain mutations ranked high/low if OST was enabled; no way to tune OST weight in scoring formula.
- Fix approach: (1) Add UI visualization of OST lDDT vs pLDDT for top mutations. (2) Make improvement_score formula configurable (currently hard-coded 0.6/0.4 split).

---

*Concerns audit: 2026-02-21*
