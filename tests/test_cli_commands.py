import re

from typer.testing import CliRunner

from protein_design_hub.cli.main import app


def _normalize_help_output(text: str) -> str:
    # Remove ANSI escape codes and collapse rich-style split tokens.
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", text)
    cleaned = cleaned.replace("-provider", "--provider").replace("-model", "--model")
    return cleaned


def test_cli_has_agents_and_pipeline_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _normalize_help_output(result.output)
    assert "agents" in out
    assert "pipeline" in out


def test_agents_run_has_provider_and_model_flags():
    runner = CliRunner()
    result = runner.invoke(app, ["agents", "run", "--help"])
    assert result.exit_code == 0
    out = _normalize_help_output(result.output)
    assert "--provider" in out
    assert "--model" in out
