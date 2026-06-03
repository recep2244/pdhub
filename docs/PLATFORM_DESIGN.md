# Protein Design Hub — Platform Design Blueprint (v4)

> **Status: decisions LOCKED (2026-06-03).** All 10 open questions from §13 are resolved in
> **§14 — Locked decisions**. Items marked *(assumption)* are sensible defaults chosen on the
> user's behalf for single-user/local v4; override any by editing §14. Predictor correction
> applied: default complex engine is **Boltz-2** (the engine actually installed), not Boltz-1.

## 1. Vision

Protein Design Hub is a **goal-driven design cockpit** that takes a scientist from a biological *intent* ("neutralize this antigen", "make this enzyme thermostable", "express this NLR in *N. benthamiana*") to a **ranked, order-ready shortlist of sequences with a defensible rationale and a fabbable construct** — wrapping the existing modeling stack (ESMFold/Chai/Boltz, ProteinMPNN/ESM-IF/RFdiffusion, ipSAE/QC/immunogenicity) so the user never has to know which tool to run in what order. The science is already strong; the value-creating work of v4 is to **re-package it around four user goals**, enforce a single *gate-vs-rank* scientific contract, attach a single *is-this-orderable?* developability gate, and harden the UI/architecture/QA discipline so the confident numbers it emits are correct.

---

## 2. Executive summary — the 8–10 biggest decisions

These are deduped across all seven role sections; where roles conflicted, the resolution is stated.

1. **Reorganize the IA around four goal-driven Tracks** (Binder, Antibody, Plant/Wheat, Mutagenesis), each a guided stepper. Demote the 15 tool-named pages to *steps* plus an "Advanced / Tools" drawer; add three missing surfaces: a **Home/Launchpad** (routes intent→Track), a first-class **Plant/Wheat Track** (most underserved persona today), and a unified **Reports/Exports** surface. The science stack stays intact — the gap is packaging, not modeling.

2. **The product's value moment is "intent → ranked shortlist + rationale + Order Pack", not "run a tool."** Every Track terminates in a downloadable **Order Pack** (ranked candidate CSV, host-specific codon-optimized FASTA with tags/signal/linkers fused, annotated construct map, assay plan, risk sheet in Ala123Lys notation). A shared context-carrying **Workbench** (viewer, compare/leaderboard, jobs, reports) persists state across steps.

3. **Enforce a hard gate-vs-rank split everywhere.** Confidence metrics (pLDDT, ipTM) **gate or tiebreak only**; ranking is always a re-normalized weighted composite over *surviving* designs (`protein_qc.composite_score` pattern), with **each signal normalized (z-score or min-max against the survivor band) before weighting** so weights are comparable across native units. **A hard-gate metric must contribute little or nothing to the survivor rank** — variance among survivors is truncated by the gate, so re-ranking on it double-counts the selection signal. **`ipSAE_min` is the primary binder/interface gate, used for ranking (its stated strength); ipTM is demoted to tiebreaker.** ipSAE thresholds are **predictor-specific and not predictor-agnostic success probabilities** — the canonical defaults (~0.6 std / ~0.7 stringent) were characterized on AF2-multimer (Dunbar & Adams 2024, biorxiv 2024.07.23.604676); they are **frozen per predictor in config and recalibrated on a labeled internal set in the golden-fixture suite** before being used as a hard gate. Until an engine is calibrated, ipSAE is used for RANKING with a looser, engine-checked gate. Block ESMFold (no PAE) from any interface decision at the registry level. Mandate design→sequence→refold self-consistency (**scTM ≥ 0.8 std / ≥ 0.9 stringent AND scRMSD ≤ 2 Å std / ≤ 1.5 Å stringent over the full designed backbone**, BindCraft/RFdiffusion convention) for binder/de-novo tracks. **Delete** the legacy pLDDT-delta mutagenesis ranking; rank by a **normalized** composite of −ΔΔG + ESM-2 zero-shot + conservation + AlphaMissense (human-only flag/tiebreak, not a primary engineering objective), pLDDT demoted to a viability gate (≥50).

4. **One Developability Gate is the platform contract.** Every sequence carries one `DevReport` with a hard rule-based `orderable` boolean; nothing reaches a candidate sheet without it. The gate is **rule-based, not composite-based** — free thiols (odd cysteine), forbidden residues, failed back-translation force NO-GO regardless of score. Collapse the three duplicate pI/GRAVY/instability implementations into one canonical core in `biophysics/properties.py`; host recommendation is a deterministic decision matrix, not prose scoring.

5. **Keep the Phosphor Lab design system (~70% right); fix the discipline.** Demote glow to focus-only, kill the `translateY` card-lift and ambient hover glow (the main "instrument vs toy" lever). Restrain aquamarine `#3fe0c5` to ≤1 element per visual group; reserve status hues strictly for QC data. Formalize the Hanken/IBM-Plex-Mono split with `tnum`+`zero` on all numbers; card radius `lg→md` (14px). Track-grouped sidebar.

6. **One confidence encoding app-wide.** The AlphaFold pLDDT ramp (`#0053d6`/`#65cbf3`/`#ffdb13`/`#ff7d45`) drives pLDDT, ipTM/pTM, PAE, and 3D coloring; promote to `ui.py` tokens and delete all ad-hoc `#1f77b4`. The very-high token `#0053d6` is a **fill-only** color (light ink on top) — it fails contrast as foreground text (see §5). PAE heatmaps use a continuous reversed confidence ramp with an **engine-aware domain**: the max-error ceiling is read from the predictor's output/metadata (AF2 emits 0–31.75 Å; Boltz-2/Chai-1 may bin differently). A fixed 0–31.75 Å domain is **AF2-specific** and is only used as a shared colorbar after non-AF2 PAE is verified rescaled to match. A **campaign funnel** (Generated→Folds→Interface→QC→Liabilities→Shortlist) is mandatory on every multi-design page, and every results page opens with one **takeaway banner** fed by `WorkflowContext.step_verdicts`. The mutation saturation heatmap uses a **separate** diverging effect scale (purple stabilizing → orange destabilizing), never the confidence ramp.

7. **WCAG 2.2 AA is a blocking release gate.** Route the raw `unsafe_allow_html` injections (**251 across 21 files as of commit `26def12`** — the CI ratchet reads the live grep count at setup time, not a hardcoded number) through typed `ui.py` helpers so ARIA/contrast/light-theme tokens are applied once; ban new raw HTML in `pages/`. Add the missing skip link (targeting Streamlit's existing main container), **best-effort** landmark roles (only what Streamlit 1.58 does not already emit — no duplicate `main`), unique per-page title and single `<h1>`. Helper-emitted a11y HTML is sanctioned raw HTML, carved out of the ratchet (§6). Status is never hue-alone. All scientific figures get alt/aria summaries plus an adjacent tabular fallback and CVD-safe colormaps.

8. **Architecture: three rings + a new `pdhub.services` seam.** `core/` imports no streamlit; `web/` imports no torch/predictor internals (enforced by import-linter). Introduce `st.cache_resource` model factory (currently **0 uses**), `st.cache_data(ttl)` for pure compute and 60s availability probes, and `@st.fragment` around every poller/viewer/scanner. Move long jobs out of the Streamlit thread into a separate `pdhub worker` process backed by the SQLite job table; heavy/batch jobs shell out to `nextflow run`. Split the 2.6k–3.9k-line pages into <400-line controllers; centralize the **~127 distinct session_state keys** (≈904 occurrences); delete the duplicate `app.py` (after a grep gate confirms nothing imports it); write a pydantic `RunManifest` to every run dir. (Counts cited as of commit `26def12`; CI reads live counts.)

9. **Make CI gates real.** Remove `|| true` on ruff/mypy and `fail_ci_if_error:false` on coverage; add per-package coverage floors (science ≥85%, total ≥75%). Build a scientific regression suite from frozen golden fixtures with per-metric tolerances plus `hypothesis` invariants (identity-mutation ΔΔG=0, antisymmetry, score ranges). Adopt a 5-tier marker-gated pyramid; the cheap 95% runs <8 min/push, GPU/external/e2e run nightly. Graceful degradation is a contract (not yet met — `energy/foldx.py` raises today): audit every external-tool wrapper to return `{status: unavailable|error}`, never raise; a break-it test drops each binary and asserts no raise.

10. **Scope assumption (resolves the recurring open question):** v4 targets **single-user / local** on Recep's RTX 4080 box. This makes the worker a simple `systemd`/docker-compose process (no Redis/SLURM broker), the Workbench backed by local SQLite + session state (no auth/sharing), and lets the wet-lab outcome loop (KPI 4) be a lightweight local "mark as ordered / mark as worked" affordance rather than an ELN/LIMS integration. Multi-user is explicitly deferred; the `services` seam keeps that door open.

---

## 3. Personas & jobs-to-be-done

| Persona | Primary JTBD | Trigger | "Done" | Sophistication |
|---|---|---|---|---|
| **Dev — De-novo Binder Designer** | Design novel binders to a target surface; hand wet-lab a small high-confidence order list | New target structure (PDB/AFDB) | 8–24 designs passing ipSAE/QC + self-consistency, with interface metrics & cost/time estimate | High ML; wants control + automation |
| **Ava — Antibody Engineer** | Turn a hit mAb/VHH into a developable, low-immunogenicity variant panel for a target epitope | Hit from screen / liability flag | 10–50 ranked variants, CDR-annotated, humanness ≥85%, PTM/aggregation/immuno cleared, synthesis-ready | High biology, low ML internals |
| **Wren — Plant/Wheat Biologist** | Engineer a plant protein (NLR, enzyme) for a trait; get a construct expressible in *N. benthamiana*/wheat | Trait hypothesis / pathogen pressure | Per-domain confidence report + codon-optimized, transit-peptide-aware construct | Deep domain biology, light protein-ML |
| **Mira — Protein Engineer (Mutagenesis)** | Find the few mutations that improve stability/activity without breaking fold or binding | Need ΔTm / ΔΔG / activity gain | Ranked mutants by ΔΔG+ESM-PLL, conservation-gated, with structural rationale + developability delta | Medium; trusts numbers, needs explanation |

**Cross-cutting JTBD all four share:** (1) "don't make me pick the tool"; (2) "tell me which ones to actually order and why"; (3) "give me an artifact I can hand to the bench / synthesis vendor."

### Data egress & confidentiality (owner: this section; new in v4)

The platform ingests **proprietary antigen/antibody/NLR sequences**. The §2.10 "single-user/local" scope does **not** by itself cover egress: today sequences can leave the box via (a) **cloud LLM "AI review" providers** (groq / cerebras / sambanova / openrouter per project memory) and (b) the **ColabFold MSA server** for AF2-multimer. This is a blocking trust gap for a pharma user (Ava). v4 contract:

- **Local-only is the default for sensitive tracks.** Default LLM provider is **local Ollama**; default complex predictor is **local Boltz-2** (no ColabFold MSA egress). AF2-multimer/ColabFold and any cloud LLM provider are **opt-in**.
- **"No sequence leaves the box unless the provider is local" toggle** in Settings/Secrets, enforced at the `services` seam: when on, any tool/provider that would transmit raw sequence off-box is disabled with an explicit reason, not silently skipped.
- **Enumerate egress per step** in the UI: each step that can transmit sequence (cloud LLM, ColabFold) shows an egress badge before it runs.
- **Input provenance in `RunManifest`:** record input **source, license, and consent** for ingested sequences, not just outputs (§10).

---

## 4. Information architecture / navigation (from scratch)

**Principle:** Top nav = *what you're trying to do* (Tracks). Tools become steps inside a Track stepper. A shared Workbench carries state across steps. Power users keep à-la-carte access via an Advanced/Tools drawer — **never gated** (guided-mode usage is a KPI, not a mandate).

**Is this actually simpler? (before/after, not just a principle)**

| | Today | Proposed v4 |
|---|---|---|
| Top-level surfaces | 15 flat pages | 4 Tracks + Workbench + Tools drawer + Agents + Home + Settings/Guide |
| Clicks: first-time guided design | open page → guess order across ≥3 pages (high cognitive load) | Home → Track → stepper (intent→track→step) |
| Clicks: power user re-running yesterday's job | 1 (straight to the page) | **must stay 1–2**: Home surfaces "recent projects / resume" and the Tools drawer reaches any single tool in 1 click — see below |

**Resolution of the "more surfaces" concern:**
- **Tools is a true escape hatch, not a maintained parallel IA.** Each Track step and its Tools-drawer twin **share one controller** (`web/views/<track>/<step>` renders the same component the Tools entry renders). There is **no second implementation** — the Tools drawer is just an ungated entry point into the same controllers. This is a hard architectural rule (Phase 4), not a convention, so "doubling maintenance" does not occur.
- **Workbench is collapsed where possible.** Compare/Leaderboard, Jobs, and Reports render as **tabs within an active run** (inside the track surface) rather than as a separate top-level area; the standalone Workbench entry exists only as a deep-link target, not as a competing destination. This avoids adding a second top-level area.
- **"Re-run yesterday's job" must be ≤2 clicks.** Home's first element is "recent projects / resume campaigns"; choosing one lands directly in that run's surface. If telemetry shows this regressing past 2 clicks, it is a release blocker for Phase 4.
- **KPI 6 (guided-mode ≥60%) acknowledges ~40% will use Tools directly** — that is acceptable *because* Tools shares controllers with the steppers (no parallel IA cost), not a sign the reorg failed.

```
Home / Launchpad        → "What do you want to design?" → routes intent to a Track
                          + recent projects, resume campaigns

TRACKS (goal-driven steppers)
  De-novo Binder         → target → design → predict → rank(ipSAE) → QC → Order Pack
  Antibody Engineering   → load Ab → CDR annot → mutate → developability → Order Pack
  Plant / Wheat Biology  → load → domain/transit annot → (design) → codon-opt → Construct
  Mutagenesis & Stability→ load → scan → ΔΔG+PLL rank → validate → Order Pack

WORKBENCH (shared, context-carrying)
  Structure & Sequence viewer · Compare/Leaderboard · Jobs & Queue · Reports & Exports

TOOLS (advanced drawer — à la carte)
  Predict · MPNN/ESM-IF/RFdiffusion · MSA · QC/metrics · Foldseek · Batch

Agents / AI Review   (LLM scientist panel — inline assist + standalone)
Settings · Guide
```

### Page mapping (keep / reorganize / drop)

| Current page | Verdict | New home |
|---|---|---|
| 14_binder | Keep, promote | Binder Track (becomes the stepper) |
| 12_antibody | Keep, promote | Antibody Track |
| 10_mutation_scanner | Keep, promote | Mutagenesis Track (primary step) |
| 0_design | Reorganize | generic design step in Tracks; plant-specific → Plant Track |
| 1_predict | Reorganize | step in every Track; also Tools drawer |
| 8_mpnn | Reorganize | "Sequence design" step + Tools drawer |
| 7_msa | Reorganize | step inside Predict/Tools |
| 4_evolution | Merge | into Mutagenesis Track (combinatorial step) |
| 2_evaluate | Reorganize | QC step + Workbench |
| 3_compare | Keep, move | Workbench → Compare/Leaderboard |
| 9_jobs | Keep, move | Workbench → Jobs |
| 11_agents | Keep, reframe | inline AI-review in Tracks + standalone Agents page |
| 5_batch | Keep, demote | Tools drawer (batch is a mode, not a goal) |
| 6_settings, 13_guide | Keep | unchanged |
| (none today) | **ADD** | Home/Launchpad, Plant/Wheat Track, Reports/Exports, Codon-opt step |

**Navigation implementation decision (resolves UX vs A11y tension):** adopt Streamlit's **native `st.navigation`/`st.Page`** (Streamlit 1.58 confirmed installed) for the top-level route graph (free AT-correct nav landmarks + `aria-current`). Two concrete Streamlit traps are addressed explicitly:

1. **Rename `web/pages/` → `web/app_pages/` (or `web/views/`) in Phase 4 and delete the numeric-prefix auto-pages.** `st.navigation` requires the page directory **not** be named `pages/` — Streamlit's legacy auto-discovery hijacks `pages/` and produces a duplicated/broken nav. The repo's pages live in `web/pages/` today; this rename is a Phase-4 work item, not a free win.
2. **Sidebar styling is best-effort, not pixel-identical.** `st.navigation`'s generated nav markup is Streamlit-owned; class names are not a stable API. Phosphor sidebar styling is layered as **CSS overrides scoped only to stable hooks (`data-testid="stSidebarNav"`)**, and a **screenshot-diff guard** catches a Streamlit upgrade that changes the nav DOM. We do **not** promise the hand-rolled Phosphor nav rendered identically on top of native navigation — visual fidelity is best-effort.

The four Tracks are **view compositions** (`web/views/<track>/`), not new science.

### Success metrics (KPIs)

| # | KPI | Target | Why |
|---|---|---|---|
| 1 | Intent→Shortlist completion rate | ≥70% | the core value moment |
| 2 | Time-to-first-shortlist | <30 min binder / <5 min mutagenesis | adoption wedge |
| 3 | Shortlist precision vs **external label** (held-out experimental outcome or external benchmark set; NOT the platform's own QC gates) | ≥90% | wet-lab trust — measured against ground truth, not self-consistency of filtering |
| 4 | Wet-lab hit rate (outcome loop) | track + improve QoQ | ground-truth metric (local "mark as worked") |
| 5 | Track adoption coverage | ~~all 4 ≥15% MAU~~ → **N/A in single-user v4**; proxy: each track exercised end-to-end in self-usage + e2e suite | validates the 4-persona bet |
| 6 | Guided-mode usage share | ~~≥60%~~ → **N/A as MAU in single-user v4**; proxy: guided path is the default entry and reaches a shortlist without touching Tools | confirms goal-IA beats tool-IA |
| 7 | Rationale-attached exports | ≥80% | the differentiator |
| 8 | Re-run/iteration rate | 2–4 healthy | detects untrusted ranking/QC |

**MAU-based KPIs (5, 6) are not measurable with a single local user (§2.10)** and are explicitly downgraded to self-usage proxies above rather than listed as measurable product metrics. A `kpi_events` eventing layer is a **day-one cross-cutting requirement**; KPIs 1–3, 7–8 are instrumentable immediately, KPI 4 via the local outcome affordance, KPIs 5–6 as self-usage/e2e proxies.

**Product-value KPIs need a validation gate, not just instrumentation.** KPI 1 (intent→shortlist ≥70%) and KPI 3 (precision vs external label ≥90%) are the bet that justifies v4; the roadmap therefore adds a **post-Phase-6 validation gate** (see §12 Phase 7) that holds the release until KPIs 1 and 3 are **met on a defined local test corpus**, not merely "instrumented."

---

## 5. Design system spec ("Phosphor Lab v4")

The four rules: **(1) one signal, not many** — aquamarine appears on ≤1 element per visual group and is never decorative; **(2) color is data, not chrome** — status hues reserved for QC verdicts; **(3) mono labels, grotesk prose, numbers are mono+tabular**; **(4) hairline over fill** — structure from 1px borders, hover is a border-color change, not a lift.

**Single source of truth for theming (resolves native-`[theme]` vs custom-CSS drift):** the native Streamlit **`[theme]`/`[theme.light]`/`[theme.dark]` in `.streamlit/config.toml` is the canonical source for every color Streamlit can theme** — background, text, primary, status, borders, code, and charts (`chartCategoricalColors`/`chartSequentialColors`). This is the supported, upgrade-safe path (Streamlit 1.58 guidance explicitly says do **not** use CSS for theming Streamlit can do natively). The `--pdhub-*` CSS variables are **restricted to what native theming cannot express**: the 5-bin confidence ramp, the `--pdhub-lookhere` highlight, and viewer chrome. Where possible the `--pdhub-*` values are **generated at runtime from `st.context.theme`** so a light/dark flip propagates automatically rather than being hand-maintained in two places. The **blocking contrast CI runs against the `config.toml` values**, not a hand-maintained CSS copy. The token tables below name colors twice only for documentation; the implementation keeps one authority per color.

### Tokens — color (base & ink)
| Token | Value | Use |
|---|---|---|
| `--pdhub-bg` | `#080b0f` | app background |
| `--pdhub-canvas` | `#0d1217` | raised canvas |
| `--pdhub-bg-card` | `#0e151b` | card/panel ink |
| `--pdhub-bg-elevated` | `#121b22` | modals, popovers, viewer chrome |
| `--pdhub-border` | `rgba(126,166,178,0.14)` | hairline |
| `--pdhub-border-strong` | `rgba(126,166,178,0.26)` | hover/active hairline |
| `--pdhub-border-focus` | `rgba(63,224,197,0.55)` | keyboard focus ring (2px) |
| `--pdhub-grid-line` | `rgba(126,166,178,0.035)` | 44px blueprint grid |

### Tokens — text (WCAG AA; contrast computed against the **`#0e151b` card** background — the surface body text actually sits on; the CI contrast script uses this exact pairing)
| Token | Value | Contrast on `#0e151b` | Use |
|---|---|---|---|
| `--pdhub-text-heading` | `#f2f7f9` | 15.8:1 | h1–h3 |
| `--pdhub-text` | `#e7eef2` | 14.9:1 | body |
| `--pdhub-text-secondary` | `#9bafbb` | 7.6:1 | subtitles |
| `--pdhub-text-muted` | `#8193a3` | 5.8:1 (6.2:1 on `#080b0f` app bg) | mono labels (do **not** regress to `#62788a`) |

> Contrast figures are recomputed/pinned by the blocking CI script against the named background; treat the values here as the expected output of that script, not hand-rounded estimates. State the background explicitly because text-muted is 5.8:1 on the card but 6.2:1 on the app bg — the gate must test the surface the text actually renders on.

### Tokens — signal, status & confidence (the only saturated hues)

**Allowed-usage column is load-bearing:** the CI contrast script tests each token against the pairing declared here (fg = used as foreground text/thin marks; fill = used only as a background with light ink on top). A fill-only token is never asserted at 4.5:1 as text.

| Token | Value | Allowed usage | Meaning |
|---|---|---|---|
| `--pdhub-primary` | `#3fe0c5` | fg + fill | aquamarine signal / interactive / pass-strong |
| `--pdhub-primary-light` | `#6bf0d8` | fg + fill | hover on signal |
| `--pdhub-primary-dark` | `#16b89c` | fill | pressed / focus fill |
| `--pdhub-on-signal` | `#04140f` | fg-on-fill | ink on filled signal |
| `--pdhub-success` | `#56d364` | fg + fill | QC pass |
| `--pdhub-warning` | `#ffb454` | fg + fill | marginal / single liability |
| `--pdhub-error` | `#ff5d6c` | fg + fill | fail / clash / aggregation |
| `--pdhub-info` | `#4cc9f0` | fg + fill | informational secondary |
| `--pdhub-conf-veryhigh` | `#0053d6` | **fill ONLY** | pLDDT≥90 / ipTM≥0.80 / PAE≤5 (fill swatch with light ink; 2.81:1 on `#0e151b` — **fails text contrast**, never used as fg text/thin marks) |
| `--pdhub-conf-veryhigh-text` | `#5a9cf0` | **fg** | text/legend/thin-swatch label for very-high confidence (≥3:1 on `#0e151b`); pairs with the `#0053d6` fill |
| `--pdhub-conf-high` | `#65cbf3` | fg + fill | pLDDT 70–90 |
| `--pdhub-conf-low` | `#ffdb13` | fg + fill | pLDDT 50–70 |
| `--pdhub-conf-verylow` | `#ff7d45` | fg + fill | pLDDT<50 / disordered |
| `--pdhub-lookhere` | `#f59e0b` | fg + fill | **reserved**: mutation/paratope "look-here" residue only |

**Resolved conflict — two oranges.** The UX reviewer wanted to unify the off-palette amber `#f59e0b` into `--pdhub-warning #ffb454` for badges/warnings/viewer. The Data-Viz reviewer reserves `#f59e0b` exclusively for "look-here" residues (mutation/paratope) and never for confidence. **Resolution:** they are two *different* semantic roles. `--pdhub-warning #ffb454` is the single source of truth for *QC-status* amber (badges, warnings, marginal-confidence residue tint). `--pdhub-lookhere #f59e0b` is a distinct token used *only* for the deliberate mutation/paratope highlight in 3D and sequence views. Neither is ad-hoc; both are tokens.

### Type scale (Hanken Grotesk + IBM Plex Mono)
| Token | Spec | Font | Use |
|---|---|---|---|
| `text-display` | 2.6rem / 800 / -0.035em | Hanken | masthead title |
| `text-h2` | 1.18rem / 700 | Hanken | section header |
| `text-h3` | 1.0rem / 600 | Hanken | card title |
| `text-body` | 0.9375rem / 1.6lh | Hanken | prose |
| `text-metric` | 1.75rem / 600 / `tnum`,`zero` | **Mono** | metric values |
| `text-label` | 0.68rem / 0.13em / UPPER | **Mono** | labels |
| `text-kicker` | 0.70rem / 0.22em / UPPER | **Mono** | instrument kicker |
| `text-code` | 0.85rem | **Mono** | sequences, IDs, FASTA |

The `Outfit` font in `apply_pro_theme` violates the spec and is replaced by Hanken/IBM Plex Mono everywhere, including Plotly figure fonts.

**Font implementation notes (not free):**
- **Self-host the woff2 in `web/static/`** and declare via `[[theme.fontFaces]]` in `config.toml`; this requires a **server restart, not hot-reload**.
- **`tnum`/`zero` only apply if the shipped woff2 build actually contains those OpenType features** — verify on a sample before relying on tabular/slashed-zero rendering (many free Hanken / IBM Plex builds include them, but it is unverified here).
- **Plotly static export ≠ browser CSS font.** Browser CSS does not reach kaleido/orca PNG export. Set `fig.update_layout(font_family=...)` **and** ensure the TTF is discoverable by kaleido; if it is not, fall back to a kaleido-bundled mono for PNGs and document the on-screen-vs-export discrepancy.

### Spacing / radius / elevation / motion
- **Spacing** (4px base): `2xs4 xs8 sm12 md16 lg24 xl32 2xl48 3xl64`. Inter-section rhythm = `2xl(48)`; intra-card = `md(16)`.
- **Radius:** `xs6 / sm10 / md14 / lg20 / full999`. Cards demote `lg→md(14)`; `lg` reserved for masthead.
- **Elevation:** `shadow-sm 0 1px 3px rgba(0,0,0,.3)` resting; `shadow-md` popover/modal; `shadow-glow 0 0 0 1px rgba(63,224,197,.25)` = **focus only**. Kill ambient hover glow.
- **Motion:** `transition-fast 0.15s`; `transition 0.22s cubic-bezier(.25,.1,.25,1)`; `fade-in 0.4s`; drop `--pdhub-bounce`. All motion respects `prefers-reduced-motion: reduce` → collapse to 0.01ms; disable spinner rotation, card hover, job-panel auto-scroll.

### Layout grid
Max width 1320px, 12-col logical grid (`st.columns`), gutter `lg(24)`. Three standard layouts: **Run** (config rail `4fr` + result canvas `8fr`), **Compare** (2–3 equal `1fr` viewers + shared metric strip), **Report** (single 760px column for transcripts/manuscript prose).

### Components
| Component | Status | Spec |
|---|---|---|
| Masthead | Refine | mono kicker + signal tick + display title; drop radial corner glow |
| Card | Refine | bg-card, hairline, radius-md, pad-lg; hover = border→strong only |
| Metric tile | Keep+ | mono value + uppercase label; add delta chip + QC verdict dot; `role="group" aria-label="{label}: {value} {unit}"` |
| Badge | Keep+ | pill ok/warn/err/info/primary; add subtle `badge-track-{ab,binder,plant,mut}` tints (badge only, not full theming — protects single-signal rule) |
| Info box | Keep | 4px left status border; maps to st.success/warning/error/info; `role="note"`, post-action error/warn `role="alert"` |
| Progress steps | Recolor | active=signal flat, completed=success; `role="list"` + `aria-current="step"`; no gradient |
| Data table | Refine | **Two distinct table types, because `st.dataframe` cannot do `scope=col` / per-cell ARIA / per-cell background via a stable API.** (a) **Large/bulk tables** (>~50 rows): native `st.dataframe` + `column_config` for mono/numeric format/width; accept `st.dataframe`'s built-in accessibility as the ceiling; ship a downloadable CSV as the tabular fallback; **drop** the per-cell ARIA/scope/tint promise here. (b) **Small shortlist/leaderboard tables** (capped at shortlist size, not 5000 rows): render a **real semantic HTML `<table>` via a sanctioned `ui.py` helper** with `<th scope=col>` + per-cell QC tint at 14% alpha + tabular AT fallback. Pick per table; never spec one component needing both. |
| Viewer frame | New, standardize | one `.pdhub-viewer` chrome; mono caption (PDB·chains·length); control cluster; residue highlight `--pdhub-lookhere`, interface `--pdhub-primary`, target neutral grey |
| Expert panel | Refine | one `pdhub-expert-card`, monogram chip (no photos), collapsible Q&A as tip info-boxes |
| pLDDT/PAE strip | Keep | canonical 5-bin confidence palette, identical across all pages |

### States
- **Empty:** dashed border + muted icon + **one primary CTA** (no dead ends).
- **Loading:** three tiers — inline spinner+mono status (<30s); progress steps with live stage (pipeline); skeleton hairline blocks (tables/viewer). Long GPU jobs show a mono elapsed-time counter.
- **Error:** info-box-error with plain-language cause + failing stage + recovery action. GPU-unavailable is a **warning** ("CPU Mode"), not an error.

**Viewer engine decision:** **3Dmol.js is the canonical embedded viewer** for v4 (fast, WebGL, the viewer-frame chrome is engine-agnostic), embedded via **`st.components.v2` / a packaged component** (not the deprecated `st.components.v1.html`). The existing PyMOL server stays for high-quality offscreen renders / screenshots only. Mol* is deferred. The 3D canvas is treated as **decorative-with-text-equivalent**: AT users get the metrics table + "download PDB / reset view / surface" Streamlit buttons rendered *outside* the iframe (`<iframe title="3D structure viewer">`). **Keyboard/Esc focus handling for the WebGL canvas is implemented in the component JS (not assumed free)** — focus must be escapable from the canvas. **The viewer is wrapped in `@st.fragment`** so a rerun elsewhere does not reload the iframe; the `?t={timestamp}` cache-buster forces a full iframe reload, which fights the <400 ms warm-rerun budget, so it fires **only on an actual structure change inside the fragment**, not on every app rerun.

---

## 6. Accessibility & performance budget

**Conformance target: WCAG 2.2 Level AA**, including the three 2.2 SC that bite dense dashboards: **2.4.11 Focus Not Obscured**, **2.5.8 Target Size (24×24)**, **3.2.6 Consistent Help**. Verify on **NVDA+Firefox** (primary) and **VoiceOver+Safari** (secondary) each release. **AA is the firm ceiling**; AAA is not pursued (the light publication theme satisfies print readability without an AAA mandate).

**Highest-value gaps (do first), framed as "best-effort landmarks given Streamlit's DOM":** Streamlit does **not** expose author-controllable *wrapping* landmarks — you cannot attach `role="main"`/`id="pdhub-main"` to Streamlit's real content container without injecting a second, non-wrapping `<div role="main">` (two `main`s = AT failure). So:
- **Audit what Streamlit 1.58 already emits** (it now ships ARIA on header/sidebar/main) and only **add what is missing** via a single top-of-app injection.
- **Skip link targets Streamlit's existing main container** (its `data-testid`, inspected and pinned), **not** a custom `#pdhub-main` that would sit beside rather than wrap real content.
- Same caveat for `banner`/`contentinfo`: add only if absent, never duplicate.
- Unique per-page `<title>` and single `<h1>` (`"<Track> · <Step> — Protein Design Hub"`) are achievable and required.

**Sanctioned-raw-HTML carve-out (resolves the contradiction with "raw-HTML → 0"):** the landmark/skip-link/ARIA HTML emitted by `ui.py` helpers (and the one top-of-app injection) is **explicitly sanctioned raw HTML** — the `pages/` raw-HTML ratchet counts only page-level injections, not these helper-emitted a11y primitives, so the a11y work does not fight the ratchet.

**ARIA centralized in `ui.py` helpers** (route the raw HTML blobs through them; ban new raw HTML in `pages/`, ratchet page-level count toward 0 in CI): metric `role=group`; badges carry text not hue-alone; steps `role=list`+`aria-current=step`; one `aria-live="polite"` job-status region per page (`assertive` only for hard failures); decorative icons/dots `aria-hidden="true"`.

**Forms:** persistent visible labels (no placeholder-as-label); sequence textareas get `aria-describedby` format help; invalid FASTA errors are programmatic text by the field. **Target size:** icon-only buttons ≥24×24. **Focus not obscured:** sticky masthead needs `scroll-margin-top` on focus targets; no `outline:none` without a replacement (stylelint).

**Light "publication" theme is a first-class contract:** pure token re-tint (no hardcoded hex in components — *this is why the raw-HTML injections must move to helpers*); in light mode the signal **text** token darkens to **`#0a7a68` (4.6:1 on white — passes AA for body text)** while `#0e8c78` (4.17:1 on white, **fails AA body, passes 3:1 large/bold/UI only**) is restricted to large/bold/UI fills; muted `#51636e` (6.25:1, fine). CI runs contrast against **both** palettes with the same fg-vs-fill allowed-usage column as dark mode, and pins the exact ratios; extend coverage to native popovers (`base="light"` in tandem); add `@media print` mapping for white-bg manuscript figures; toggle is keyboard-operable, persisted, announced.

### Performance budget (Streamlit, RTX 4080 baseline)
| Metric | Budget | Enforcement |
|---|---|---|
| Warm rerun (cached) | **<400 ms** | timing log |
| Cold page load | <2.5 s | — |
| GPU/registry/structure probes per rerun | **0** (cached) | grep gate |
| CSS payload | **1× per session** (flag in session_state) | code rule |
| Raw `unsafe_allow_html` per page (page-level only; helper-emitted a11y HTML excluded) | trend→0 from live baseline (251 / 21 files @ `26def12`) | CI count must not increase |
| Inline base64 images | ≤200 KB each | review |
| Fonts | **self-host** Hanken/IBM-Plex/Material-Symbols; remove 3 render-blocking Google `@import` | change |
| Largest raw dataframe (bulk `st.dataframe` type only) | ≤5000 rows | paginate; semantic-HTML shortlist tables capped at shortlist size |

**Mandatory patterns:** `@st.fragment` around every poller/viewer/scanner-filter (**exactly 1 real `@st.fragment` today** — in `web/app.py`; other grep hits for "fragment" are unrelated biology code, so this is a near-greenfield lever and the single biggest perf win); `st.cache_resource` for models/PyMOL worker/Ollama client; `st.cache_data` for parsed PDBs/MSAs/registry/score tables; lazy-import torch/pymol/foldseek inside functions only; gate slider-driven scans behind a Run button or fragment.

---

## 7. Data-viz & storytelling standards

**Rule zero: a quantity gets exactly one visual encoding across the entire app.**

- **One confidence ramp** (the four `conf-*` tokens above) drives pLDDT, ipTM/pTM, PAE, and 3D coloring; delete every ad-hoc `#1f77b4`. PAE heatmap = same ramp **reversed and continuous**, with an **engine-aware max-error domain** read from the predictor metadata (AF2 = 0–31.75 Å; Boltz-2/Chai-1 read their own ceiling). Square aspect, chain-break lines in muted grey. When comparing PAE across engines on one colorbar, rescale non-AF2 PAE to a common domain first and label it AF2-equivalent; do **not** silently clamp every engine to 0–31.75 Å (it visually compresses non-AF2 inter-domain/interface error). ipTM/pTM render as a band-colored confidence chip + value, never a lone number. Threshold bands drawn via `add_hrect` at 0.15 opacity. One `confidence_colorbar()` feeds all surfaces.
- **Amino-acid property palette** (categorical, CVD-safe, orthogonal to the continuous confidence ramp): hydrophobic `#f2c14e`, polar `#8fd694`, positive `#5aa9e6`, negative `#ef6f6c`, aromatic `#c39bd3`, Gly `#cfd8dc`, Pro `#b39ddb`, Cys `#ffd166`+outline. Cysteine and N-glyc sequons carry a **glyph overlay** (color alone is insufficient).
- **Mutation saturation heatmap** uses a **separate diverging effect scale**, NOT the confidence ramp: **purple = stabilizing ← white → orange = destabilizing**, keyed to the **ΔΔG+ESM-PLL composite** (matching the mutagenesis rework). WT cell gets a black border; gaps/unscored are hatched grey, never colored zero.
- **Campaign funnel is mandatory** on every multi-design page (Generated → Folds → Interface → QC → Liabilities → Shortlist), each bar labeled with absolute count and % retained — it turns "I made 1000 designs" into "8 are worth ordering."
- **Ranking scatter** plots each track's two decision axes: binder = ipSAE × pLDDT; antibody = developability × humanness; mutagenesis = ΔΔG × ESM-PLL. Shortlisted points get the aquamarine ring; the rest are muted.
- **Takeaway banner** opens every results page (one sentence + driving number + status border), fed by `step_verdicts` so the LLM verdict and the visual story are the same object. Examples: 1_predict "This fold is [trustworthy/partly disordered]" (mean pLDDT + %≥70); 14_binder "[K] of [N] binders are order-worthy" (funnel shortlist count); 10_mutation_scanner "Position Ala123 is the [best/worst] move" (top ΔΔG+PLL hit).
- **3D color-by menu order fixed everywhere:** `pLDDT · chain · secondary-structure · spectrum`. Target chain always neutral grey; designed/binder chain aquamarine; mutated/paratope residue `--pdhub-lookhere` sticks+sphere; interface contacts as sticks within 5 Å.
- **Universal annotation rules:** every confidence object ships an inline legend; three-letter Ala123Lys notation in all labels/tooltips; units on every axis (PAE Å, ΔΔG kcal/mol, Tm °C); threshold lines labeled with the caveat ("CDR-H3 pLDDT <70 is normal"); color never the sole channel.
- **Static PNG export path is in scope** (publication / BIOS6380-style on white): the same encoding renders against the light theme; Plotly→kaleido byte-stable for fixed data.
- **Property-palette toggle (resolved):** ship the single fixed property-class scheme as default; expose a *power-user toggle* (property-class / hydrophobicity-gradient / conservation) only inside the MSA and antibody views, not globally.

---

## 8. Scientific workflows — the four tracks

**Cross-cutting principles:** (1) gates are hard filters, ranking is a re-normalized composite over survivors only; (2) predictor-appropriate metrics — **ESMFold has no PAE, blocked for any interface decision**; (3) self-consistency mandatory for generated backbones (breaks circularity); (4) composites surface `n_terms` and grey out when <3 levels present; (5) report pass-rate, never a lone absolute score as decision-grade.

**Default complex predictor (resolved):** **Boltz-2 is the default decision-grade complex engine** on the RTX 4080 (open, no API key, emits PAE for ipSAE). It **fits 12 GB only up to a token ceiling** — complex VRAM scales with total token count, so large antibody Fv+antigen or multidomain NLR complexes (the Plant track explicitly targets these) **will OOM**. Degradation path: above the configured per-engine token ceiling, fall back to CPU/ColabFold or chunked PAE; OOM is a first-class `capability()` reason and a §5 error state ("complex too large for 12 GB — using fallback"). **Chai-1** is the offered alternative; **AF2-multimer** via ColabFold where installed. ESMFold is never selectable for the complex step.

### Track 1 — De-novo binder
target+hotspot → **RFdiffusion** backbone → **ProteinMPNN** sequence design → Boltz/Chai complex re-prediction → gate → composite rank → order top-N.

> **Scope note (no over-claim):** v4 binder generation uses the modules that exist in the repo today — **RFdiffusion + ProteinMPNN** (`design/{rfdiffusion,proteinmpnn,esmif}`). **BindCraft and LigandMPNN are NOT in the codebase** and are therefore **out of v4 scope**; they are net-new tool integrations (own venv/install/availability work), tracked as a Phase-4 exit line item if pulled in — this is not "packaging existing modeling."

All thresholds below are **engine-calibrated defaults frozen per predictor in config** (Boltz-2 default), recalibrated against the golden-fixture labeled set; they are not predictor-agnostic physical lines.

| Stage | Metric | Gate (std→stringent) | Role |
|---|---|---|---|
| Monomer self-consistency | scRMSD (full backbone) **and** scTM | scRMSD ≤2.0→1.5 Å **and** scTM ≥0.8→0.9 | hard gate |
| Monomer confidence | pLDDT | ≥85→90 | hard gate |
| Interface confidence | **ipSAE_min** (engine-calibrated) | **~0.6→~0.7** (frozen per predictor) | **primary gate + ranking** |
| Interface quality (pDockQ) | pDockQ *(if computed)* | ≥0.23 acceptable → ≥0.50 confident | supporting gate |
| Interface quality (LIS) | LIS *(if computed)* | engine-calibrated ≥0.3 (own scale) | supporting gate |
| Interface energetics | ΔG — **pick ONE engine** | Rosetta ≤−10→−15 **REU**; *or* FoldX ≤−7→−10 **kcal/mol** | gate **or** rank term (not both) |
| Packing | shape complementarity | ≥0.50→0.60 | rank term |
| Developability | instability<40, GRAVY<0.4, odd-Cys=0 | — | hard gate |

**Self-consistency** is over the full designed backbone (motif/interface-restricted scRMSD reported separately for interface-grafted designs). scTM ≥ 0.8 / 0.9 follows BindCraft/RFdiffusion self-consistency practice — scTM 0.5 is "same fold," not a validated self-consistent design, so it is **not** the gate.

**pDockQ and LIS are different metrics on different scales** (pDockQ acceptable ~0.23 / confident ~0.5; LIS its own 0–1 construction) and are gated independently on whichever is computed for the active engine — never with one shared threshold pair.

**Rosetta interface energy (REU) ≠ FoldX interaction energy (kcal/mol).** Pick one engine for the ΔG gate; REU only loosely maps to kcal/mol. The chosen ΔG engine is used as **either** a gate **or** a rank term, not both, to avoid double-counting.

**Gate-vs-rank principle applied:** ipSAE is the primary hard gate **and** its strongest independent use is ranking, but because every survivor has already cleared the ipSAE floor, the **independent** terms drive the rank. Normalized composite over survivors:

Rank = `0.20·ipSAE_norm + 0.25·(−ΔG)_norm + 0.20·SC_norm + 0.20·scTM_norm + 0.15·ESM2-PLL_norm` — each term z-scored against the survivor band before weighting; ipSAE is **down-weighted (0.20, was 0.30)** since variance among survivors is truncated by its gate. **ipTM = tiebreaker only.** Order top 8–24 ipSAE-passing designs spanning ≥3 backbones. Single-point GBSA/contact-energy is **qualitative sign-only, never a rank term**.

### Track 2 — Antibody engineering
VH/VL/VHH → CDR annotation (Chothia/IMGT/Kabat) → Fv/complex prediction → interface + developability + immunogenicity gates → composite rank.

- **CDR-H3 pLDDT ≥70 is normal** — do NOT gate H3 at 85.
- Antigen interface ipSAE (engine-calibrated ~0.6, frozen per predictor) gates **only if antigen structure available**.
- **Liabilities are position-dependent**: re-weight PTM/aggregation by **CDR vs framework** location; never present a flat motif count as the antibody risk score. CDR PTMs critical, framework minor.
- **Two distinct composite formulas — never mix interface-gated and developability-only candidates in one ranked table.** Each rank is tagged with the mode that produced it.
  - **Antigen-present mode:** Rank = `0.25·(1−immuno) + 0.25·(1−PTM_CDR) + 0.20·ipSAE_norm + 0.15·humanness + 0.15·Tm_norm` (all terms normalized before weighting).
  - **Developability-only mode (no antigen, per Q4):** Rank = `0.30·(1−immuno) + 0.30·(1−PTM_CDR) + 0.20·humanness + 0.20·Tm_norm` — **no interface term**; do not substitute a surrogate "affinity" under the ipSAE weight.
- Humanness gate ≥85%. Advance only humanness-passing, CDR-PTM-clean candidates; **flag, don't auto-mutate**.

### Track 3 — Plant / wheat biology
gene/protein → domain architecture (`nlr_domains`) → structure prediction → per-domain QC → expression-context (transit peptide, codon, PTM) → decision.

- **Interpretive, not generative** — the deliverable is a per-domain confidence report + inter-domain PAE block map, not a ranking of 1000 designs.
- **Order Pack still applies, reconciling §2.2.** Plant's Order Pack is **the codon-optimized construct itself** (transit-peptide-aware, host codon table), *not* a ranked shortlist. **`DevReport.orderable` still gates it** via the deterministic chemistry checks (back-translation, forbidden residues, restriction sites) even though AlphaMissense and the survivor ranking are disabled. Wren's "Done" = an expressible construct, which is an orderable artifact.
- **Never report a single global pLDDT** on multidomain proteins (flexible linkers drag the mean down); report **per-domain pLDDT + inter-domain PAE** and gate on domain-level confidence.
- Localization via `transit_peptide`; codon optimization for host. **AlphaMissense is NOT applied** (undefined for plant/synthetic sequences).
- **Degraded path (common): no MSA / no AlphaMissense.** Plant sequences frequently lack a usable MSA. The per-domain report must render with single-sequence confidence only, explicitly labeling conservation/coevolution signals as unavailable rather than silently scoring zero, and the construct (orderability via deterministic chemistry) still ships.

### Track 4 — Mutagenesis / variant effect
WT + mutation set/saturation scan → multi-signal scoring → fold-preservation check → composite rank.

**Every signal is normalized to a common scale before weighting** (the same re-normalized-composite rule §2/§8 mandate elsewhere). Native units/ranges differ wildly: ESM-2 zero-shot is a log-likelihood ratio (~−10…+5, often negative); AlphaMissense is a 0–1 pathogenicity probability; ΔΔG is kcal/mol (~−5…+5); conservation and fold-preservation are on their own scales. Raw-weight multiplication without normalization is meaningless, so each column below lists its **native range** and is **z-scored (or min-max'd) per scan** before the weight applies.

| Signal | Source | Native range | Direction | Weight (on normalized signal) |
|---|---|---|---|---|
| Stability ΔΔG | FoldX/heuristic | kcal/mol, ~−5…+5 | −ΔΔG beneficial | 0.30 |
| Fold preservation | OST lDDT/RMSD vs WT | lDDT 0–1 / RMSD Å | preserved positive | 0.30 |
| Evolutionary fitness | ESM-2 zero-shot | LLR, ~−10…+5 | higher tolerated | 0.25 |
| Conservation | MSA family | bits / freq 0–1 | conserved→penalize | 0.15 |
| Pathogenicity | AlphaMissense (**human-only**) | prob 0–1 | **flag/tiebreak only** | — (not in composite) |
| Confidence | Δmean pLDDT | pLDDT 0–100 | **gate only** | — (not in composite) |

**AlphaMissense is demoted to a flag/tiebreak, not a composite term.** A "pathogenic" human-disease missense call is a directional clinical prior, *not* an engineering fitness objective ("bad for the patient" ≠ "bad for stability/activity"); it is surfaced as an advisory flag for human targets only and used only to break ties among otherwise-equal candidates.

Viability gate: mutant mean pLDDT ≥50 (hard kill, not a rank term — confidence does not enter the composite). **DELETE the legacy `0.6·Δmean_pLDDT + 0.4·Δlocal_pLDDT` ranking everywhere** — it rewards mutations that merely make ESMFold more confident, uncorrelated with stability.

**Platform-wide weak practices to fix:** pLDDT as fitness/affinity; ESMFold for interfaces; skipped self-consistency; global pLDDT on multidomain; flat liability counts as antibody risk; GBSA sold as affinity; thin composites compared to full ones.

---

## 9. Wet-lab / developability requirements

**First principle:** nothing leaves the platform without a `DevReport`:
```
DevReport(sequence_id) -> { verdict: GO|CONDITIONAL|NO-GO, tier_scores{...},
  flags[Liability(pos in Ala123Lys)], host_recommendation{host,format,tags,signal_peptide},
  orderable: bool, construct: ConstructSpec|None }
orderable = verdict != "NO-GO" and no unrejected hard-stop flag
```

**Kill the duplication (real surface area):** `biophysics/properties.py` becomes the single source for pI/GRAVY/instability/ε/MW/aliphatic. The duplication is far wider than two files — pI/GRAVY/instability-style helpers are re-implemented in **~15 call sites**: `analysis/{protein_utils,esm_dimer,mutation_scanner,sequence_metrics,protein_qc}.py`, `analysis/wet_lab_advisor.py`, `cli/commands/design.py`, `agents/mutagenesis_agents.py`, `evolution/fitness_landscape.py`, `web/scientific_context.py`, `web/pages/{1_predict,8_mpnn,12_antibody}.py`, and `core/config.py`. Phase 3 exit requires an **import-linter / grep gate** asserting these properties are imported only from `biophysics/properties.py` and never re-implemented. Reconcile the two conflicting threshold tables into one tiered (lenient/standard/strict) table:

| Metric | Nature | lenient | standard | strict | Hard-stop |
|---|---|---|---|---|---|
| Instability | heuristic, **advisory** | <55 | <45 | <40 | no (warn) |
| GRAVY | deterministic | <0.4 | <0.25 | <0.0 | no |
| pI distance from assay pH | deterministic | >0.5 | >1.0 | >1.5 | no |
| Aggregation (APR count) | **uncalibrated heuristic, advisory** | ≤3 | ≤2 | ≤1 | no (warn only) |
| Aggregation/solubility **relative score** | **uncalibrated heuristic, advisory** | ≥0.30 | ≥0.45 | ≥0.60 | no (warn only) |
| Unpaired Cys (odd) | deterministic chemistry | warn | fail | fail | **yes (free thiol)** |
| N-glyco sequons | deterministic | report | report | host-gated | host-dep |
| Deamidation NG/NS | deterministic motif | ≤4 | ≤2 | 0 in CDR/active site | no |
| Forbidden residues (X,*,stop) | deterministic chemistry | fail | fail | fail | **yes** |
| Back-translation feasibility | deterministic | fail | fail | fail | **yes** |
| Length vs host ceiling | deterministic | host | host | host | host-dep |

**Hard-stops are reserved for deterministic chemistry only** (free thiol, forbidden residues, failed back-translation). **Heuristic-derived metrics never hard-stop** — the in-house aggregation/solubility predictor is a **"TANGO/CamSol-style" reimplementation built to avoid new deps and is uncalibrated**; TANGO and CamSol are separately-validated methods and this is not them, so a "0.45" or "APR ≤2" is a relative house cutoff, not a validated threshold. The "predicted solubility" field is therefore renamed **"relative solubility score"** with **no probability semantics**, and gates on it are **advisory warnings** (CONDITIONAL at most), never NO-GO, until validated against a labeled set. The composite is informational; **the orderability gate is rule-based and deterministic** — a perfect composite with a free thiol is NO-GO.

**Host recommender is a deterministic decision matrix** (returns first feasible host + fallbacks), keyed on computed SS-bond count, glyco sequons, GRAVY, MW — not prose scoring. Defaults: **antibody → HEK/CHO + Protein A**; **binder → *E. coli*** (periplasm if SS bonds); **plant → *N. benthamiana*** + plant codon table + transit-peptide handling. Hosts: *E. coli* cyto/peri, HEK293, CHO, *P. pastoris*, *N. benthamiana*, cell-free.

**Assay binding:** ipSAE/ipTM/ΔG → SPR (K_D, k_off); screening → BLI; Tm/instability/ΔΔG → DSF/nanoDSF; aggregation/solubility/pI → SEC(-HPLC)/MALS; charge variants → cIEF. **DSF Tm + SEC %monomer are, by convention, the cheapest experiments that kill the most bad designs — gate expensive SPR/BLI behind expression + DSF + SEC.**

These pass/fail bars are **configurable house defaults, modality-specific — not universal biophysical truths**: the developable Tm bar depends on modality and formulation (default **Tm > 55 °C**; **antibodies want Tm/Tm-onset > 65–70 °C**; many useful binders/enzymes function below 55 °C), and **%monomer ≥ 95%** is a reasonable house cutoff, not a law. They live in config per track/modality so a viable design is not rejected on a borderline convention.

**Order Pack (per track, downloadable):** ranked candidate CSV (`rank,id,sequence,verdict,composite,K_D_pred,Tm_pred,%monomer_pred,host,format,flags(Ala123Lys),orderable`, sorted GO→CONDITIONAL→NO-GO); host-specific codon-optimized FASTA with tags/signal/linkers fused; annotated construct map; assay plan (DSF→SEC→SPR/BLI ladder, reuse `_build_purification_strategy`); risk sheet in three-letter notation with specific mitigations.

**Missing checks to add:** restriction-site/forbidden-motif scan (BsaI/BsmBI/NdeI/XhoI), APR/aggregation predictor, predicted-solubility gate, and a **one-click PTM-acknowledgement gate** so users cannot silently ship a CDR-H3 NG motif. Mutagenesis track must surface a **developability *delta* vs WT** so a stabilizing mutation that introduces an aggregation patch is flagged.

**Resolved defaults:** pI-distance gate targets **PBS 7.4 by default, ~6.0 for the mAb/antibody track** (formulation pH). Codon-optimized FASTA export defaults to **Twist** paste-ready format (IDT/GenScript selectable). Aggregation/solubility predictor is a **lightweight built-in heuristic** (TANGO/CamSol-style, no new deps) — **uncalibrated and advisory only** (never a hard-stop; see threshold table); a validated external predictor is a later enhancement. Mutagenesis developability-delta is a **warning, not a hard gate** (stability/activity is the user's goal; surface the trade-off, don't block).

---

## 10. Architecture, reproducibility & state/caching

**Three rings, one rule:** `core/` never imports `streamlit`; `web/` never imports `torch`/predictor internals. Enforced by import-linter in CI.

| Ring | Package | May import | Must NOT import |
|---|---|---|---|
| Domain | `pdhub.core` (types, config, registry, manifest) | stdlib, pydantic | streamlit, torch |
| Science | predictors/design/evaluation/analysis/biophysics/evolution/msa/energy | core | streamlit, other tools' internals |
| Orchestration | **`pdhub.services` (NEW)** — job submit, run-dir layout, manifest, availability, result loading | core + science | streamlit |
| Agents | `pdhub.agents` | core + services (read-only) | streamlit, science internals |
| Interface | `web/` + `cli/` (thin) | core + services + agents | torch, predictor classes, `is_installed()` |

**Concrete moves:** create `pdhub/services/` as the single seam web+CLI+Nextflow all call; delete the duplicate `src/protein_design_hub/app.py` (keep `web/app.py`); split monster pages to <400-line controllers with rendering in `web/views/<track>/` and compute in `services`; the four tracks are view compositions, no science forks.

**Caching (two-tier):** `st.cache_resource` (currently 0 uses — biggest latency win) for ESM2/ESMFold/MPNN weights, foldseek DB handle, Ollama client, PyMOL worker; `st.cache_data(ttl,max_entries, content-hash keys)` for PLL scores, metric tables, parsed CIF→DataFrame; `st.cache_data(ttl=60)` for availability probes. Cache at the `services` boundary (science stays pure). **Cache keys are content hashes, not file paths.**

**State:** namespace the **~127 distinct session_state keys** (≈904 total read/write occurrences as of `26def12` — the task is namespacing the ~127 *unique* keys, not 904 sites) as `f"{track}.{page}.{field}"`, centralize defaults in `web/state.py` (`init_state()`, typed getters). A **key-alias shim** maps legacy → namespaced keys during the transition so in-flight sessions and bookmarks do not break.

**Long jobs out of the Streamlit thread:** keep SQLite job table (`~/.pdhub/jobs.db`) as source of truth; run a **separate `pdhub worker` process** (systemd / docker-compose) — survives reruns/multi-tab/restart. `core/job_manager.py` exists but **the worker process, the pure-Python fallback batch executor, and a docker-compose worker service are net-new builds** with real concurrency risk: the job table is read by Streamlit and written by the worker (and possibly Nextflow) simultaneously, so the table runs in **WAL mode with a single-writer worker** (or advisory locking) to avoid SQLite write-contention/locking. Light analysis inline+cached; single prediction → queue, poll via `@st.fragment(run_every="2s")`; campaigns/batch → `nextflow run` spawned detached, worker tails the trace.

**Reproducibility:** one pydantic `RunManifest` written to every run dir by both app and Nextflow paths (`run_id`, UTC ts, git SHA, pip-freeze hash, tool versions, GPU model, **resolved seeds**, full resolved config, input content hashes, **input provenance {source, license, consent}**, and **egress record {which steps/providers transmitted sequence off-box}** per §3); `seed=None` flags the run "non-reproducible". Run-dir layout `outputs/<run_id>/{manifest.json, inputs/, predictions/, evaluation/, logs/, nextflow/}`; migrate the 143 unstructured dirs. Keep the Nextflow CLI-shellout pattern, add a `slurm` profile, keep `PREDICT{maxForks=1}`, surface timeline/report/trace from the Jobs page.

**Resolved infra questions:** v4 is single-user/local → worker is a plain systemd/compose process, **no Redis/SLURM broker**, no per-user run-dir isolation. **Nextflow stays optional**: a pure-Python fallback executor handles batch/campaign for users without Nextflow/Java; Nextflow is the preferred path when present (gives free provenance). Add a retention/GC policy with a size cap on `outputs/` (default: keep last N runs + manual pin) to prevent disk fill.

**Config & secrets:** type API keys as `pydantic.SecretStr`; split `Settings` (serialized into manifest) from `Secrets` (never serialized); precedence defaults→`config/default.yaml`→`~/.pdhub/config.yaml`→env `PDHUB_*`→session override; secrets only from env/`.env`/`st.secrets`. Keep the 12-provider preset table and the qwen2.5:14b Ollama default.

**Availability gating:** one `pdhub.services.availability.capability(tool) -> {installed,version,gpu_required,gpu_present,venv,reason}`, 60s-cached; pages query this, never `torch.cuda`/`is_installed()` directly. Unavailable tools render disabled with an inline install command. Keep per-tool venvs (`.venv_esm3`, `.venv_immunebuilder`); ship `Dockerfile.gpu` (CUDA) distinct from the slim CPU image, selected by Nextflow profile.

---

## 11. Quality & verification strategy

**Make CI gates real:** drop `|| true` on ruff/mypy (blocking on changed files; legacy mypy grandfathered via baseline); drop `fail_ci_if_error:false`; add coverage floors **total ≥75%, science packages (evaluation/analysis/biophysics) ≥85%**. Gitignore committed artifacts (`.coverage`, `coverage.xml`, `molprobity.out`, `.venv_*`, `outputs/`).

**Five-tier marker-gated pyramid** (every push runs unit+science+integration+smoke, target **<8 min**; gpu/external/e2e deselected by default):

| Tier | Marker | Scope | Budget | Runs |
|---|---|---|---|---|
| Unit | `unit` | pure functions (metrics, qc, scoring, ipsae) | <30s | push |
| Scientific golden | `science` | frozen inputs→expected outputs + tolerances | <60s | push |
| Integration | `integration` | orchestrator/campaign/CLI/IO, stubbed predictors | <3min | push |
| Page smoke | `smoke` | bare-script exec (keep) + Streamlit AppTest, all 15 pages | <2min | push |
| E2E | `e2e` | Playwright, 1 flow/track | <15min | nightly+release |
| Heavy | `gpu`,`external` | real ESMFold/ESM2/Foldseek/FoldX/OpenMM | unbounded | nightly self-hosted |

**Scientific regression (the heart):** golden numeric fixtures in `tests/data/golden/` with decision-grade tolerances (ipSAE abs 1e-4; TM-score abs 0.02; liabilities exact int; composite abs 1e-6; mutation rank **Spearman ≥ 0.999 with explicit tie tolerance** — *not* exact 1.0, because ties in the ΔΔG+ESM-PLL composite are common in saturation scans and make exact rank order non-deterministic; pair with a **stable sort + documented deterministic tiebreak key** stated alongside the seed, and assert exact rank equality only on the de-duplicated top-K; immunogenicity ±5; ESM2/AlphaMissense **sign+ranking** exact). `hypothesis` invariants: identity-mutation ΔΔG=0, antisymmetry, metric ranges ([0,100]/[0,1]/PAE≥0), composite key-order invariance, self-consistency scTM bounded in [0,1] (the *gate* is scTM≥0.8/0.9, not an invariant). Three tiny pinned PDBs (56-aa monomer, 2-chain complex, antibody Fv). Golden updates require `--update-golden` + reviewer sign-off acknowledging the scientific delta.

**Graceful degradation is a contract — and is partly aspirational today, not just a generalization.** The Foldseek wrapper already conforms, but **`energy/foldx.py` currently raises `EvaluationError('foldx', 'FoldX executable not found')` in ≥4 places** (and other predictor/Rosetta/OpenMM wrappers are unaudited), so making "every external-tool wrapper returns `{status: unavailable|error}`, never raises" true is a **work item, not a free generalization**. FoldX is a rank/gate term in Track 1, so its non-conformance directly violates the contract today. **Exit criterion (Phase 0 audit + Phase 2/3 fixes):** audit all wrappers (foldx, rosetta, openmm, predictors) to return status dicts; a **"break-it" test drops each binary and asserts no raise.**

**Visual & a11y gates:** WCAG contrast of design tokens is **blocking** (scripted, both dark+light palettes — aquamarine-on-ink is high-risk); PyMOL non-black-JPEG check (>5 KB) in smoke; full axe-core + screenshot diffs nightly.

**Per-track Definition of Done** (each row checkable in CI or manual checklist): CDR annotation exact vs reference Fv + humanness/MHC-II ±5 (antibody); ipSAE/ipTM/pDockQ within tol + protein-qc keeps good/drops bad+odd-Cys (binder); transit/NLR/codon-CAI match golden (plant); ranking Spearman ≥0.999 (tie-tolerant, top-K exact) vs golden + conservation gating + Ala123Lys rendered (mutagenesis). Cross-cutting: every new science function ships golden/invariant test + docstring with units+citation + graceful-degradation + ≥85% module coverage.

**Verification discipline:** each phase ends with a tracked `docs/qa/phase-N.md` Verification Report; phase-exit requires a **"break-it-on-purpose"** check (mutate a sign, drop a predictor) proving the gates have teeth. A green default CI is necessary but not sufficient.

**Resolved QA scope:** the RTX 4080 laptop **is** the self-hosted nightly GPU runner; gpu/external/e2e are non-blocking nightly (a red nightly blocks the next release, not day-to-day merges). 3–4 small reference PDBs + a handful of ClinVar variants are committed (size/license OK at this scale). Per-push blocking budget is **<8 min**; if exceeded, integration moves to a parallel CI job.

---

## 12. Phased implementation roadmap

Each phase has a goal + exit criteria mapped to the decisions in §2.

**Phase 0 — Foundations & teeth (decisions 8, 9).**
Goal: make the skeleton honest before adding features. Create `pdhub.services`; **grep-gate that nothing imports `src/protein_design_hub/app.py` before deleting it** (keep `web/app.py`); import-linter contract; `st.cache_resource` model factory + availability `capability()`; remove `|| true`, add coverage floors; gitignore artifacts; wire the 5-tier pyramid; `kpi_events` table; **audit all external-tool wrappers (foldx raises today, rosetta, openmm, predictors) to return status dicts**; **session_state key-alias shim** + one-time **run-dir migration script (with `--dry-run`)** for the 143 existing dirs.
Exit: all pages smoke-pass; CI gates **fail loudly on a seeded bad commit**; warm rerun <400 ms on one cached page; 0 GPU probes per rerun on that page; break-it test drops the FoldX binary and asserts **no raise**; run-dir migration dry-run reported; `docs/qa/phase-0.md` signed.

**Phase 1 — Design-system & a11y contract (decisions 5, 6, 7).**
Goal: lock the token contract and route HTML through helpers. Phosphor Lab v4 tokens (confidence ramp + two-orange resolution); flatten glow/hover; self-host fonts; skip link + landmarks + per-page title/h1; migrate raw HTML into `ui.py` helpers (ratchet begins); confidence/property colormap helpers + `confidence_colorbar()`; `takeaway_banner()`.
Exit: axe-core 0 critical/serious on all pages; contrast suite green on **both** palettes; raw-HTML count strictly decreasing; NVDA skip-link+landmark pass; reduced-motion verified.

**Phase 2 — Scientific contract (decision 3).**
Goal: gate-vs-rank everywhere. Block ESMFold for interface steps at the registry; ipSAE primary binder gate, ipTM→tiebreaker; self-consistency gate for binder/de-novo; **delete** legacy pLDDT-delta mutagenesis ranking; per-domain pLDDT + inter-domain PAE for multidomain; surface `n_terms`/grey-out thin composites.
Exit: golden + invariant suites green (mutation rank Spearman ≥0.999 tie-tolerant; ipSAE within 1e-4); "break-it" sign-flip caught; Boltz wired as default complex engine (with token-ceiling/OOM fallback).

**Phase 3 — Developability gate & Order Pack (decision 4).**
Goal: one orderable contract. Collapse pI/GRAVY/instability into `properties.py`; tiered threshold table; rule-based `DevReport.orderable`; deterministic host matrix; add restriction-site scan, APR/solubility heuristic, PTM-acknowledgement gate; Order Pack export (CSV + Twist FASTA + construct map + assay plan + risk sheet).
Exit: liability golden fixtures green; free-thiol forces NO-GO in tests; Order Pack downloads for a sample sequence in each track.

**Phase 4 — Tracks & Workbench (decisions 1, 2).**
Goal: ship the goal-driven IA. **Rename `web/pages/` → `web/app_pages/` and remove numeric-prefix auto-pages** (required for `st.navigation`); `st.navigation`/`st.Page` route graph + track-grouped sidebar scoped to `data-testid="stSidebarNav"` + screenshot-diff guard; Home/Launchpad intent router (recent/resume ≤2 clicks); the four Track steppers as `web/views/<track>/` sharing one controller with the Tools-drawer twins; Workbench tabs-within-a-run (no second top-level area); Plant/Wheat Track first-class; Tools drawer preserves à-la-carte access. **If BindCraft/LigandMPNN are pulled into scope, add their net-new install/venv/availability work as an explicit exit line item** (default v4 binder track is RFdiffusion+ProteinMPNN only).
Exit: each track reaches an exported shortlist end-to-end (KPI 1 instrumented); per-track DoD rows green; "re-run yesterday's job" ≤2 clicks verified; Tools entry and stepper step proven to call the same controller (no duplicate impl); guided-mode self-usage proxy green.

**Phase 5 — Jobs, reproducibility & outcome loop (decisions 8, 10).**
Goal: durable runs + provenance + KPI 4. **Out-of-process `pdhub worker` and the pure-Python fallback executor are net-new components** (not just config) — the SQLite job table is now touched by Streamlit + worker + Nextflow concurrently, so **enable WAL mode + a single-writer worker (or advisory locking)** to avoid write-contention; `RunManifest` (incl. input provenance + egress record) on every run; run-dir layout migration + retention/GC; Nextflow optional with pure-Python fallback; "mark as ordered / mark as worked" local affordance.
Exit: a 20-min ESMFold run survives a Streamlit rerun; **worker + pure-Python fallback each have a smoke test; a concurrent Streamlit-read + worker-write does not lock the SQLite job table**; manifest reconstructs a run; nightly GPU/e2e green; KPI 4 capturing data.

**Phase 6 — Polish & publication theme (decisions 6, 7).**
Goal: manuscript-grade output. Light publication theme full coverage + `@media print`; static PNG/kaleido export; raw-HTML count → 0 in `pages/`; campaign funnel + ranking scatter on every multi-design page; screenshot-diff baseline.
Exit: print one results page to white-bg PDF with CVD-safe figures; raw-HTML in `pages/` = 0 (page-level; sanctioned helper a11y HTML excluded); full pyramid + nightly green; release sign-off on all four DoDs.

**Phase 7 — Product-value validation gate (KPIs 1, 3).**
Goal: prove the v4 bet, not just instrument it. Assemble a **defined local test corpus** (held-out targets/variants with known/external outcomes or an external benchmark set) and measure **KPI 1 (intent→shortlist ≥70%)** and **KPI 3 (shortlist precision vs external label ≥90%)** end-to-end.
Exit (**release is held until met**): KPI 1 ≥70% and KPI 3 ≥90% on the corpus; KPIs 5–6 reported as self-usage/e2e proxies (MAU N/A in single-user v4); a written go/no-go in `docs/qa/phase-7.md`. The roadmap is **not "done" until this gate passes** — engineering-green Phases 0–6 are necessary but not sufficient.

---

## 13. Open questions for the user

> **All resolved — see §14 (Locked decisions).** Retained below for traceability. Items #2, #9, #10 were closed with flagged defaults that the user can override.

Most panel open-questions are resolved inline above under the single-user/local v4 scope. These remain genuine product/scope calls:

1. **Outcome loop depth (KPI 4):** is the lightweight local "mark as ordered / mark as worked" affordance sufficient for v4, or do you want eventual ELN/LIMS integration on the roadmap?
2. **Plant/Wheat host priority:** which host(s) to build codon-opt + expression guidance for *first* — *N. benthamiana* (assumed default), wheat, or also *E. coli*/HEK for plant-protein expression? This scopes the first Track build.
3. **LLM agent panel positioning:** required "AI review" gate inside each Track (affects trust + latency) or strictly optional inline assist? (Assumed optional inline for v4 — confirm.)
4. **Antibody antigen availability:** is an antigen structure usually on hand (enabling Fv–antigen ipSAE), or should the antibody Track default to developability-only ranking without an interface gate?
5. **Composite-weight recalibration:** freeze the published composite weights for v1, or build the SPR/BLI/Tm feedback loop to retune them over time (depends on Q1)?
6. **Retention policy specifics:** how many runs / what disk cap for `outputs/` before GC kicks in?
7. **Confirm the two defaults chosen above:** Boltz-2 as default complex engine, and Twist as default synthesis-FASTA format — change either?
8. **Data egress / cloud providers (new):** is local-only (Ollama + Boltz, no ColabFold MSA egress) an acceptable default for sensitive tracks, with cloud LLM / ColabFold strictly opt-in behind the "no sequence leaves the box" toggle? Are any cloud providers acceptable for non-proprietary inputs?
9. **ipSAE recalibration corpus (new):** do you have (or can you assemble) a labeled internal binder set per predictor (Boltz-2 especially) to recalibrate the ipSAE gate, or should v4 ship Boltz with the AF2-derived ~0.6/~0.7 default flagged "uncalibrated, ranking-only" until such data exists?
10. **Validation corpus for Phase 7 (new):** what is the local test corpus for KPI 1/3 — held-out experimental outcomes, an external benchmark set, or both? Without it the release-holding validation gate cannot run.

---

## 14. Locked decisions (v4)

Resolutions to the §13 open questions, decided 2026-06-03 for **single-user / local v4** on the
RTX 4080 box. `(assumption)` = a default chosen on the user's behalf; change here to override.

| # | Question | **Decision** | Rationale / consequence |
|---|----------|--------------|--------------------------|
| 1 | Outcome loop depth (KPI 4) | **Lightweight local "mark as ordered / mark as worked"** affordance; ELN/LIMS deferred to a later milestone. | Feeds the `kpi_events` table; no external integration burden in v4. Enables the Phase-5 outcome loop without a vendor dependency. |
| 2 | Plant/Wheat host priority | **_N. benthamiana_ first**, then wheat (stable), with **E. coli** as generic fallback. *(assumption)* | Matches existing transit-peptide / Agrobacterium / NLR tooling; transient agroinfiltration is the fastest validation path. Codon-opt tables + signal/transit-peptide logic built for *N. benthamiana* in Phase 3/4. |
| 3 | LLM agent panel positioning | **Optional inline assist, opt-in** — never a required gate. | Preserves the existing "step" vs "llm" pipeline modes; avoids adding LLM latency/trust to the critical path. Agent verdicts annotate, never block. |
| 4 | Antibody antigen availability | **Developability-only ranking by default**; enable **Fv–antigen ipSAE interface gate only when an antigen structure is supplied.** | Two documented composite formulas (§8 Track 2). No interface gate is invented when there's no antigen. |
| 5 | Composite-weight recalibration | **Freeze published weights for v1**; build the SPR/BLI/Tm recalibration loop later (depends on #1, Phase 5+). | Reproducible, citable v1 ranking. Weights live in config so a future labeled set can retune them. |
| 6 | `outputs/` retention / GC | **Keep last 50 runs OR 20 GB (whichever first)**, manifest-aware GC; **never GC a run flagged "ordered."** User-configurable in Settings. | Bounds disk on the laptop; protects wet-lab-relevant runs. Implemented in Phase 5 with the run-dir migration. |
| 7 | Default complex engine / synthesis format | **Boltz-2** default complex engine (Chai-1 alternative; AF2-multimer/ColabFold where installed); **Twist** default synthesis FASTA (IDT/GenScript switchable). | Boltz-2 is installed, open, local-GPU, emits PAE for ipSAE. Order Pack FASTA matches Twist's paste-ready format by default. |
| 8 | Data egress / confidentiality | **Local-only default** (Ollama LLM + local Boltz-2/ESMFold, no ColabFold MSA egress). Cloud LLM / ColabFold **strictly opt-in** behind a **"no sequence leaves the box"** toggle; egress recorded in `RunManifest`. | Sensitive sequences never leave the machine unless explicitly allowed; provenance is auditable. |
| 9 | ipSAE recalibration corpus | **No labeled internal set assumed** → ship Boltz-2 ipSAE with the **AF2-derived ~0.6/0.7 default flagged "uncalibrated — ranking-only"**; used for ranking + a looser engine-checked gate until a labeled set exists. *(assumption)* | Avoids a false predictor-agnostic success-probability claim. Becomes a hard gate per engine once the golden-fixture labeled set is built. |
| 10 | Phase-7 validation corpus | **External benchmark set** (held-out known binders/variants) for KPI 1/3; gate also accepts **internal experimental outcomes** when the user supplies them. *(assumption)* | The release-holding validation gate can run without internal lab data; tightens automatically as #1's outcome loop accumulates labels. |

**Net effect on the roadmap:** all phases are unblocked. The only items still dependent on the
user are *data they alone possess* — a labeled binder set (#9) and experimental outcomes / a chosen
benchmark (#10) — and v4 ships sensible "uncalibrated/external" defaults until those arrive, so no
phase is blocked waiting on them. Plant host (#2) can be re-pointed before Phase 4 at zero cost.

---

## Review notes

This revision applies the three reviewers' findings (Scientific rigor; Design & accessibility feasibility; Completeness & over-claim), all of which carried a *needs-work* verdict. Summary of changes:

**Blockers (all resolved):**
- §5 added a fg-vs-fill **allowed-usage column**; `--pdhub-conf-veryhigh #0053d6` is now **fill-only** (2.81:1 fails text contrast) with a new `--pdhub-conf-veryhigh-text #5a9cf0` (≥3:1) for fg text/legends.
- §6 light theme: signal **text** token darkened to `#0a7a68` (4.6:1 on white); `#0e8c78` (4.17:1, fails AA body) restricted to large/bold/UI fills.
- §4 `st.navigation`: stated the required **`web/pages/` → `web/app_pages/` rename**, removal of numeric auto-pages, sidebar styling scoped to `data-testid="stSidebarNav"` + screenshot-diff guard, and dropped the "pixel-identical Phosphor nav" promise.

**Major (resolved):**
- §2/§8 **ipSAE thresholds** reframed as predictor-specific, frozen-per-engine, recalibrated in the golden suite; cited Dunbar & Adams (AF2-multimer); split gate vs rank so ipSAE is **down-weighted (0.30→0.20)** in the binder composite.
- §8 self-consistency gate tightened to **scTM ≥0.8/0.9 AND scRMSD ≤2.0/1.5 Å** (BindCraft/RFdiffusion), full-backbone.
- §6/§7 PAE domain made **engine-aware** (no app-wide 0–31.75 Å clamp).
- §8 Track 4 weights: all signals **normalized before weighting** with native ranges stated; **AlphaMissense demoted to flag/tiebreak** (not a composite term).
- §9 heuristic solubility/APR marked **advisory, never hard-stop**; "predicted solubility" → "relative solubility score" (no probability semantics); hard-stops reserved for deterministic chemistry.
- §3 added a **Data egress & confidentiality** subsection (local-only default, "no sequence leaves the box" toggle, input provenance in `RunManifest`).
- §8 Track 1 **BindCraft/LigandMPNN removed from v4 scope** (not in repo); binder track = RFdiffusion+ProteinMPNN.
- §10/§11 **FoldX raise** acknowledged as a work item with a break-it test; graceful-degradation reworded as audit, not generalization.
- §10 counts corrected: **~127 distinct session_state keys** (≈904 occurrences); **1 real `@st.fragment`** today; **251 raw-HTML / 21 files**; CI reads live counts at `26def12`.
- §3/§9 duplication enumerated across **~15 call sites** with an import-linter/grep gate.
- §4 IA: added **before/after click/surface table**, made **Tools a shared-controller escape hatch** (no parallel IA), Workbench collapsed to tabs-within-a-run, "re-run yesterday" ≤2 clicks.
- §5/§6 reconciled **native `[theme]` (canonical) vs `--pdhub-*` CSS** (restricted to what native theming can't express; contrast CI runs against `config.toml`).
- §6 **landmarks** reframed as best-effort given Streamlit's DOM (no duplicate `main`; skip link targets Streamlit's container; helper a11y HTML carved out of the raw-HTML ratchet).
- §5 **data table** split into bulk `st.dataframe` (drop per-cell ARIA promise + CSV fallback) vs small semantic-HTML shortlist tables.
- §12 added **Phase 7 product-value validation gate** (holds release until KPI 1 ≥70% / KPI 3 ≥90% on a defined corpus); MAU KPIs 5/6 downgraded to self-usage proxies (single-user).

**Minor (resolved):**
- §8 split **LIS vs pDockQ** into separate rows with their own thresholds (pDockQ ~0.23/0.5); separated **Rosetta REU vs FoldX kcal/mol** and made ΔG gate-or-rank, not both.
- §4 KPI 3 redefined against an **external label** (not the platform's own QC).
- §8 Track 2 **two distinct composite formulas** (antigen-present vs developability-only); no slashed `affinity/ipSAE` fallback.
- §9 **Tm/%monomer** framed as configurable modality-specific house defaults (antibody Tm 65–70 °C).
- §8 Boltz "fits 12 GB" qualified with a **token ceiling + OOM fallback**.
- §8 Track 3 Plant **Order Pack reconciled** (codon-optimized construct, `orderable` still gates) + **no-MSA degraded path**.
- §11 mutation rank **Spearman ≥0.999 tie-tolerant** (not exact 1.0) with stable-sort tiebreak.
- §5 font notes: self-host woff2 + `[[theme.fontFaces]]` + restart; verify `tnum`/`zero`; kaleido font registration for Plotly PNG export.
- §5 viewer: **`st.components.v2`**, explicit Esc/keyboard handling, **`@st.fragment`** wrap + cache-buster only on structure change.
- §5/§6 contrast figures recomputed against the named `#0e151b` card background and pinned via CI.
- §0/§5/§10 migration shim + run-dir `--dry-run` script + grep gate before deleting the duplicate `app.py`.

**Judgment calls made (flagged for the user):**
- Binder composite re-weighted to `0.20·ipSAE + 0.25·(−ΔG) + 0.20·SC + 0.20·scTM + 0.15·ESM2-PLL` (chose option (a): keep ipSAE as gate, down-weight in rank, let independent terms drive).
- AlphaMissense **removed from the mutagenesis composite entirely** (demoted to flag/tiebreak) rather than kept at a justified weight — it conflates clinical pathogenicity with engineering fitness.
- FoldX stringent ΔG set to ≤−7/−10 kcal/mol (illustrative; needs FoldX-specific calibration — confirm or replace with empirical values).
- `--pdhub-conf-veryhigh-text #5a9cf0` and light `#0a7a68` are proposed tints meeting the contrast bar; exact hex is the designer's call as long as the pinned ratio holds.

**Deferred minor issues (open):**
- Exact recalibrated ipSAE / FoldX-ΔG numeric thresholds per engine are left to the golden-fixture calibration step (Q9) rather than guessed in the doc.
- The precise `data-testid` selectors for Streamlit 1.58's main container and sidebar nav must be inspected at implementation time (Phase 1/4) — named here as a method, not a literal value.
- Whether `tnum`/`zero` ship in the chosen woff2 build is unverified until the font files are selected.

---

*This blueprint supersedes ad-hoc per-page conventions. Token/variable names (`--pdhub-*`) are frozen per project memory; the four Tracks, the gate-vs-rank rule, the single Developability Gate, and the one confidence encoding are the load-bearing contracts every contributor must honor.*
