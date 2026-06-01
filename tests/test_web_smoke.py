import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "src" / "protein_design_hub" / "web"
APP_SCRIPT = WEB_ROOT / "app.py"
PAGE_SCRIPTS = sorted((WEB_ROOT / "pages").glob("*.py"))


def _run_script_smoke(script: Path) -> None:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-40:])
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        raise AssertionError(
            f"Script failed: {script}\n"
            f"stdout tail:\n{stdout_tail}\n\nstderr tail:\n{stderr_tail}"
        )


def test_web_app_smoke_loads():
    _run_script_smoke(APP_SCRIPT)


@pytest.mark.parametrize("script", PAGE_SCRIPTS, ids=lambda p: p.name)
def test_web_page_smoke_loads(script: Path):
    _run_script_smoke(script)
