# Testing Patterns

**Analysis Date:** 2026-02-21

## Test Framework

**Runner:**
- Pytest 7.0.0+ (configured in `pyproject.toml`)
- Config file: No separate `pytest.ini`; settings in `pyproject.toml` [tool.pytest] or inherited defaults
- Auto-discovery: files matching `test_*.py` in `tests/` directory

**Assertion Library:**
- Standard Python `assert` statements (Pytest rewrites assertions for detailed failure messages)
- No external assertion library; built-in assertions sufficient

**Run Commands:**
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output with test names
pytest -x                 # Stop on first failure
pytest -k pattern         # Run tests matching pattern
pytest --cov             # Coverage report (pytest-cov installed)
pytest tests/test_*.py   # Run specific test file(s)
```

## Test File Organization

**Location:**
- Tests co-located in `tests/` directory at project root (separate from source)
- One test file per module or feature area

**Naming:**
- Test files: `test_*.py` (e.g., `test_agent_pipeline_integrity.py`, `test_energy_metrics.py`)
- Test functions: `test_*()` (e.g., `test_step_pipeline_has_5_computational_steps()`)
- Test classes (optional): `Test*` for grouping related tests (rarely used, prefer flat functions)

**Structure:**
```
tests/
├── test_agent_pipeline_integrity.py    # Agent orchestration, verdict parsing
├── test_cli_commands.py                # CLI command structure
├── test_energy_metrics.py              # Evaluation metrics (clash, contact energy, etc.)
├── test_llm_config.py                  # Configuration resolution
├── test_mutation_scanner_ost.py        # Mutation scanner with OST metrics
├── test_optional_integrations.py       # Optional dependency handling
├── test_sequence_metrics.py            # Sequence-level metrics
└── test_web_smoke.py                   # Web app smoke tests
```

## Test Structure

**Suite Organization:**

Tests use flat function structure with Pytest fixtures. No class-based suites currently.

```python
# Pattern: test file organization

def test_basic_functionality():
    """Test a single behavior."""
    # Setup
    orchestrator = AgentOrchestrator(mode="step")

    # Assertion
    assert len(orchestrator.agents) == 5

def test_with_fixture(tmp_path):
    """Test using Pytest fixtures."""
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)

    metric = ClashScoreMetric(cutoff_angstrom=2.0)
    result = metric.compute(pdb)

    assert "clash_score" in result

@pytest.mark.parametrize("script", PAGE_SCRIPTS, ids=lambda p: p.name)
def test_parametrized(script):
    """Test multiple inputs."""
    # Test runs once per item in PAGE_SCRIPTS
```

**Examples from codebase:**

From `test_agent_pipeline_integrity.py`:
```python
def test_step_pipeline_has_5_computational_steps():
    orchestrator = AgentOrchestrator(mode="step")
    steps = orchestrator.describe_pipeline()
    assert len(steps) == 5
    assert [s["name"] for s in steps] == [
        "input",
        "prediction",
        "evaluation",
        "comparison",
        "report",
    ]

def test_llm_pipeline_has_12_integrated_steps():
    orchestrator = AgentOrchestrator(mode="llm")
    steps = orchestrator.describe_pipeline()
    names = [s["name"] for s in steps]
    assert len(steps) >= 10

    required_order = [
        "input",
        "llm_planning",
        "prediction",
        ...
    ]
    indices = [names.index(name) for name in required_order]
    assert indices == sorted(indices)
```

**Patterns:**
- **Setup:** Create objects/fixtures needed for test
- **Action:** Call the function/method being tested
- **Assertion:** Use `assert` to verify expected outcome
- **Cleanup:** Pytest handles via fixtures (e.g., `tmp_path` auto-deleted)

## Mocking

**Framework:**
- Built-in `unittest.mock` via standard library (not imported directly in tests)
- Pytest `monkeypatch` fixture for patching module attributes and environment vars

**Patterns:**

From `test_mutation_scanner_ost.py`:
```python
def test_mutation_scanner_adds_ost_comprehensive_metrics(monkeypatch, tmp_path):
    import protein_design_hub.evaluation.composite as composite

    class DummyEvaluator:
        def __init__(self, metrics):
            self.metrics = metrics

        def evaluate(self, model_path, reference_path=None):
            return SimpleNamespace(metadata={})

        def evaluate_comprehensive(self, model_path, reference_path):
            return {"global": {"lddt": 0.91, "rmsd_ca": 1.25}}

    monkeypatch.setattr(composite, "CompositeEvaluator", DummyEvaluator)

    scanner = MutationScanner(
        predictor="esmfold_api",
        evaluation_metrics=[],
        run_openstructure_comprehensive=True,
    )
    result = scanner.calculate_biophysical_metrics(model, ref, evaluation_metrics=[])
    assert result["extra_metrics"]["ost_comprehensive"]["global"]["lddt"] == 0.91
```

**What to Mock:**
- External dependencies (evaluator classes, LLM APIs, filesystem operations)
- Classes with expensive initialization or side effects
- Patching is done via `monkeypatch.setattr(module, "ClassName", MockClass)`

**What NOT to Mock:**
- Core logic being tested
- Dataclasses and simple value objects
- Pydantic models used for input validation
- Agent pipeline orchestration (test integration instead)

## Fixtures and Factories

**Test Data:**

From `test_energy_metrics.py` (inline constants):
```python
MINI_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
...
TER
END
"""

def test_clash_score_metric(tmp_path: Path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)
    # Test runs with temp file
```

**Built-in Pytest Fixtures Used:**
- `tmp_path`: Temporary directory for test files (auto-cleaned)
- `monkeypatch`: Patch module attributes and environment vars
- `capsys`: Capture stdout/stderr (not shown in examples but available)

**Location:**
- Test data constants: Defined at top of test file (MINI_PDB, etc.)
- Temporary files: Via `tmp_path` fixture passed to test function
- No conftest.py file; fixtures defined locally or via Pytest built-ins

## Coverage

**Requirements:**
- Not enforced by CI (no minimum coverage threshold configured)
- Generated via `pytest --cov` (pytest-cov plugin)

**View Coverage:**
```bash
pytest --cov=protein_design_hub --cov-report=html
# Generates htmlcov/index.html with line-by-line coverage
```

**Coverage artifacts:**
- `.coverage` file in project root (binary coverage data)
- `coverage.xml` in project root (XML report format)

## Test Types

**Unit Tests:**
- **Scope:** Individual functions, classes, agents
- **Approach:** Setup minimal state, call single function, assert output
- **Examples:**
  - `test_step_pipeline_has_5_computational_steps()` — tests orchestrator pipeline assembly
  - `test_compute_sequence_metrics_basic()` — tests sequence metric computation
  - `test_clash_score_metric()` — tests single evaluation metric

**Integration Tests:**
- **Scope:** Multi-component interactions (agents, evaluators, config)
- **Approach:** Create real objects, orchestrate workflow, verify end state
- **Examples:**
  - `test_llm_pipeline_has_12_integrated_steps()` — verifies full LLM pipeline construction
  - `test_composite_evaluator_wires_energy_fields()` — verifies metrics compose correctly
  - `test_llm_config_resolve_env_key()` — verifies config reads environment

**E2E/Smoke Tests:**
- **Framework:** Subprocess calls (not Pytest framework-level)
- **Scope:** Full application load (CLI, web pages)
- **Approach:** Run script via subprocess, check exit code and output
- **Example:** `test_web_app_smoke_loads()` in `test_web_smoke.py`
  ```python
  def _run_script_smoke(script: Path) -> None:
      proc = subprocess.run(
          [sys.executable, str(script)],
          cwd=str(ROOT),
          capture_output=True,
          timeout=90,
      )
      if proc.returncode != 0:
          raise AssertionError(f"Script failed: {script}\n...")

  def test_web_app_smoke_loads():
      _run_script_smoke(APP_SCRIPT)
  ```

## Common Patterns

**Async Testing:**
- No async/await in codebase currently
- All agent functions are synchronous
- LLM calls via OpenAI client are blocking (synchronous)

**Error Testing:**

From `test_agent_pipeline_integrity.py` (custom test agent):
```python
class _FailVerdictAgent(BaseAgent):
    name = "llm_evaluation_review"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.step_verdicts["evaluation_review"] = {
            "status": "FAIL",
            "key_findings": ["test failure verdict"],
        }
        return AgentResult.ok(context, "fail verdict emitted")

class _TailAgent(BaseAgent):
    name = "report"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.extra["tail_ran"] = True
        return AgentResult.ok(context, "tail")

# Then orchestrate with custom agents to verify failure handling
```

**Verdict Parsing Testing:**
```python
def test_parse_verdict_uses_structured_contract():
    summary = (
        "Findings...\n"
        'VERDICT_JSON: {"step":"evaluation_review","status":"FAIL",'
        '"key_findings":["clash score too high"],'
        '"thresholds":{"clash_score":"< 20"},'
        '"actions":["run refinement"]}'
    )
    verdict = _parse_verdict_from_summary(summary, "evaluation_review")
    assert verdict["status"] == "FAIL"
    assert verdict["key_findings"] == ["clash score too high"]
    assert verdict["source"] == "verdict_json"
```

**Environment Variable Testing:**
```python
def test_llm_config_resolve_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMConfig(provider="openai", base_url="", model="", api_key="")
    resolved = cfg.resolve()
    assert resolved.api_key == "sk-test"
```

**Parametrized Testing:**
```python
@pytest.mark.parametrize("script", PAGE_SCRIPTS, ids=lambda p: p.name)
def test_web_page_smoke_loads(script: Path):
    _run_script_smoke(script)
    # Runs once for each file in PAGE_SCRIPTS, with custom id
```

## Test Execution

**Development workflow:**
```bash
# Run all tests
pytest

# Run tests matching name pattern
pytest -k "pipeline"

# Run specific file
pytest tests/test_agent_pipeline_integrity.py

# Verbose with full output
pytest -v

# Stop on first failure
pytest -x
```

**CI Integration:**
- Pre-commit hooks run: Black, Ruff, MyPy, but not tests
- GitHub Actions (if configured) would run pytest as part of CI
- No test-specific pytest.ini; config via pyproject.toml

## Known Test Limitations

**Coverage gaps:**
- Web UI pages tested via smoke tests (load check only)
- LLM calls not tested (mocked in mutation scanner test, but no real integration test)
- Actual predictor tools (ColabFold, ESMFold, etc.) not mocked in tests
- Async/concurrent agent execution not tested (agents run sequentially)

**Slow tests:**
- Smoke tests timeout at 90s (for web page load)
- Energy metrics tests slow due to BioPython operations on synthetic PDB files

---

*Testing analysis: 2026-02-21*
