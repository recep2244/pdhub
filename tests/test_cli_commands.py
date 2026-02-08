from typer.testing import CliRunner

from protein_design_hub.cli.main import app


def test_cli_has_agents_and_pipeline_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "agents" in result.output
    assert "pipeline" in result.output


def test_agents_run_has_provider_and_model_flags():
    runner = CliRunner()
    result = runner.invoke(app, ["agents", "run", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.output
    assert "--model" in result.output
