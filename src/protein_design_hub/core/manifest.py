"""Run provenance manifests for Protein Design Hub.

A :class:`RunManifest` captures everything needed to reproduce and audit a
pipeline run: a stable run identifier, the wall-clock creation time (always
passed in by the caller so this module never touches the clock at import or
construction time), the track that produced it, the provenance of every input
(source + content hash), best-effort tool versions, the parameters used, an
egress policy (local-only by default, with optional destinations) and the
output paths produced.

The model prefers ``pydantic`` when it is importable so manifests get
validation for free; when pydantic is absent it transparently falls back to a
dataclass-based implementation exposing the same public surface
(``to_dict`` / ``to_json`` / ``save`` / ``load`` and the ``Egress`` /
``InputProvenance`` helpers). Either way no heavy/optional dependency is
required, so this module imports cleanly in a minimal CI environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "InputProvenance",
    "Egress",
    "RunManifest",
    "PYDANTIC_AVAILABLE",
    "MANIFEST_FILENAME",
]

#: Default filename used by :meth:`RunManifest.save` / :meth:`RunManifest.load`.
MANIFEST_FILENAME = "run_manifest.json"


try:  # pragma: no cover - exercised by whichever branch is installed
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    PYDANTIC_AVAILABLE = False


def _normalize_paths(paths: Any) -> list[str]:
    """Coerce an iterable of path-likes into a list of plain strings."""
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(p) for p in paths]


if PYDANTIC_AVAILABLE:

    class InputProvenance(BaseModel):
        """Provenance record for a single run input.

        ``source`` describes where the input came from (a path, URL, accession
        or free-text label) and ``hash`` is a content fingerprint (e.g. a
        ``sha256`` hex digest) used to detect drift between runs.
        """

        source: str
        hash: Optional[str] = None
        kind: Optional[str] = None

        def to_dict(self) -> dict:
            return self.model_dump()

    class Egress(BaseModel):
        """Egress policy for a run.

        ``enabled`` is ``False`` by default, meaning the run is local-only and
        no data leaves the machine. When ``enabled`` is ``True``,
        ``destinations`` lists the remote sinks data may be sent to.
        """

        enabled: bool = False
        destinations: list[str] = Field(default_factory=list)

        def to_dict(self) -> dict:
            return self.model_dump()

    class RunManifest(BaseModel):
        """Reproducibility/provenance manifest for a single pipeline run."""

        run_id: str
        created_at: str
        track: str
        inputs: dict[str, InputProvenance] = Field(default_factory=dict)
        tools: dict[str, str] = Field(default_factory=dict)
        params: dict[str, Any] = Field(default_factory=dict)
        egress: Egress = Field(default_factory=Egress)
        outputs: list[str] = Field(default_factory=list)
        metadata: dict[str, Any] = Field(default_factory=dict)

        def to_dict(self) -> dict:
            """Return a JSON-serialisable plain ``dict``."""
            return self.model_dump()

        def to_json(self, *, indent: int = 2) -> str:
            """Serialise the manifest to a JSON string."""
            return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

        @classmethod
        def from_dict(cls, data: dict) -> "RunManifest":
            """Construct a manifest from a plain ``dict`` (inverse of ``to_dict``)."""
            return cls.model_validate(data)

else:

    from dataclasses import asdict, dataclass, field

    @dataclass
    class InputProvenance:
        """Provenance record for a single run input (dataclass fallback)."""

        source: str
        hash: Optional[str] = None
        kind: Optional[str] = None

        def to_dict(self) -> dict:
            return asdict(self)

    @dataclass
    class Egress:
        """Egress policy for a run (dataclass fallback)."""

        enabled: bool = False
        destinations: list[str] = field(default_factory=list)

        def to_dict(self) -> dict:
            return asdict(self)

    @dataclass
    class RunManifest:
        """Reproducibility/provenance manifest for a single pipeline run."""

        run_id: str
        created_at: str
        track: str
        inputs: dict[str, InputProvenance] = field(default_factory=dict)
        tools: dict[str, str] = field(default_factory=dict)
        params: dict[str, Any] = field(default_factory=dict)
        egress: Egress = field(default_factory=Egress)
        outputs: list[str] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            # Allow callers to pass plain dicts / lists and normalise them so
            # the public API matches the pydantic branch exactly.
            self.outputs = _normalize_paths(self.outputs)
            norm_inputs: dict[str, InputProvenance] = {}
            for key, value in (self.inputs or {}).items():
                if isinstance(value, InputProvenance):
                    norm_inputs[key] = value
                else:
                    norm_inputs[key] = InputProvenance(**value)
            self.inputs = norm_inputs
            if isinstance(self.egress, dict):
                self.egress = Egress(**self.egress)

        def to_dict(self) -> dict:
            """Return a JSON-serialisable plain ``dict``."""
            return {
                "run_id": self.run_id,
                "created_at": self.created_at,
                "track": self.track,
                "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
                "tools": dict(self.tools),
                "params": dict(self.params),
                "egress": self.egress.to_dict(),
                "outputs": list(self.outputs),
                "metadata": dict(self.metadata),
            }

        def to_json(self, *, indent: int = 2) -> str:
            """Serialise the manifest to a JSON string."""
            return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

        @classmethod
        def from_dict(cls, data: dict) -> "RunManifest":
            """Construct a manifest from a plain ``dict`` (inverse of ``to_dict``)."""
            return cls(**data)


# ---------------------------------------------------------------------------
# save / load helpers (shared behaviour, attached to both implementations)
# ---------------------------------------------------------------------------


def _manifest_save(self: "RunManifest", directory) -> Path:
    """Write the manifest to ``<directory>/run_manifest.json`` and return the path."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_FILENAME
    path.write_text(self.to_json(), encoding="utf-8")
    return path


def _manifest_load(cls, path) -> "RunManifest":
    """Load a manifest from a JSON file (or a directory containing one)."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    data = json.loads(p.read_text(encoding="utf-8"))
    return cls.from_dict(data)


# Attach as instance/class methods so the API is uniform across both branches.
RunManifest.save = _manifest_save  # type: ignore[attr-defined]
RunManifest.load = classmethod(_manifest_load)  # type: ignore[attr-defined]
