"""
PyMolAI-compatible MCP server — exposes `run_pymol_command` and
`capture_viewer_snapshot` over the MCP stdio transport.

Run as a subprocess by claude-agent-sdk (McpStdioServerConfig).
PYMOL_PORT env-var selects which PyMOL HTTP server to target.
"""

import os
import base64
import httpx
from mcp.server.fastmcp import FastMCP

_PORT = int(os.environ.get("PYMOL_PORT", "8592"))
_BASE = f"http://localhost:{_PORT}"

mcp = FastMCP("PyMolAI")


@mcp.tool()
def run_pymol_command(command: str) -> str:
    """Execute an arbitrary PyMOL command on the loaded structure.

    Returns 'OK' when the command succeeded (viewer is updated),
    or an error string if it failed.  Use standard PyMOL syntax, e.g.:
      color red, chain A
      show surface
      hide everything
      set_view (…)
    """
    try:
        r = httpx.post(
            f"{_BASE}/api/run_command",
            json={"command": command},
            timeout=30,
        )
        if r.status_code == 200:
            ct = r.headers.get("content-type", "")
            if "image" in ct:
                return "OK — view updated"
            return r.json().get("result", "OK")
        return f"Error {r.status_code}: {r.text[:200]}"
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def capture_viewer_snapshot() -> str:
    """Capture the current PyMOL viewport as a JPEG image.

    Returns a base64-encoded data-URI string you can display or analyse.
    Call this after issuing commands to verify the visual result.
    """
    try:
        r = httpx.get(f"{_BASE}/api/frame", timeout=30)
        if r.status_code == 200:
            b64 = base64.b64encode(r.content).decode()
            return f"data:image/jpeg;base64,{b64}"
        return f"Error {r.status_code}: could not capture snapshot"
    except Exception as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
