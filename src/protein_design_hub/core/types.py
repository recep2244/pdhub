"""Core data types for Protein Design Hub."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class PredictorType(str, Enum):
    """Available predictor types."""

    COLABFOLD = "colabfold"
    CHAI1 = "chai1"
    BOLTZ2 = "boltz2"
    ESMFOLD = "esmfold"
    ESMFOLD_API = "esmfold_api"


class MoleculeType(str, Enum):
    """Molecule types for multi-molecule prediction."""

    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    LIGAND = "ligand"


class MetricType(str, Enum):
    """Available evaluation metrics."""

    LDDT = "lddt"
    TM_SCORE = "tm_score"
    QS_SCORE = "qs_score"
    RMSD = "rmsd"
    PLDDT = "plddt"
    CLASH_SCORE = "clash_score"
    CONTACT_ENERGY = "contact_energy"
    ROSETTA_ENERGY = "rosetta_energy"
    SASA = "sasa"
    INTERFACE_BSA = "interface_bsa"
    SALT_BRIDGES = "salt_bridges"
    OPENMM_GBSA = "openmm_gbsa"
    ROSETTA_SCORE_JD2 = "rosetta_score_jd2"
    FOLDX_DDG = "foldx_ddg"


@dataclass
class Sequence:
    """A biological sequence."""

    id: str
    sequence: str
    molecule_type: MoleculeType = MoleculeType.PROTEIN
    description: str = ""

    def __len__(self) -> int:
        return len(self.sequence)

    def __post_init__(self):
        # Validate sequence
        self.sequence = self.sequence.upper().strip()
        if self.molecule_type == MoleculeType.PROTEIN:
            valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        elif self.molecule_type in (MoleculeType.DNA, MoleculeType.RNA):
            valid_chars = set("ACGTU")
        else:
            valid_chars = None

        if valid_chars:
            invalid = set(self.sequence) - valid_chars
            if invalid:
                raise ValueError(f"Invalid characters in sequence: {invalid}")


@dataclass
class MSA:
    """Multiple Sequence Alignment."""

    query_id: str
    sequences: list[tuple[str, str]]  # List of (header, sequence) tuples
    format: str = "a3m"

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def alignment_length(self) -> int:
        if self.sequences:
            return len(self.sequences[0][1])
        return 0


@dataclass
class Template:
    """Structure template for prediction."""

    id: str
    path: Path
    chain_id: str = "A"
    sequence_identity: Optional[float] = None

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Template file not found: {self.path}")


@dataclass
class Constraint:
    """Distance or contact constraint for prediction."""

    residue1: int
    residue2: int
    distance_min: float = 0.0
    distance_max: float = 8.0
    chain1: str = "A"
    chain2: str = "A"
    weight: float = 1.0


@dataclass
class PredictionInput:
    """Input data for structure prediction."""

    job_id: str
    sequences: list[Sequence]
    msa: Optional[MSA] = None
    templates: list[Template] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    num_models: int = 5
    num_recycles: int = 3
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir:
            self.output_dir = Path(self.output_dir)

    @property
    def total_length(self) -> int:
        return sum(len(seq) for seq in self.sequences)

    @property
    def is_multimer(self) -> bool:
        return len(self.sequences) > 1


@dataclass
class StructureScore:
    """Scores for a predicted structure."""

    plddt: Optional[float] = None
    plddt_per_residue: Optional[list[float]] = None
    ptm: Optional[float] = None
    iptm: Optional[float] = None
    pae: Optional[list[list[float]]] = None
    confidence: Optional[float] = None


@dataclass
class PredictionResult:
    """Result from a structure prediction."""

    job_id: str
    predictor: PredictorType
    structure_paths: list[Path]
    scores: list[StructureScore]
    runtime_seconds: float
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    @property
    def best_structure(self) -> Optional[Path]:
        """Return the best structure by confidence score."""
        if not self.structure_paths:
            return None
        if not self.scores:
            return self.structure_paths[0]

        best_idx = 0
        best_score = -1.0
        for i, score in enumerate(self.scores):
            confidence = score.confidence or score.plddt or 0.0
            if confidence > best_score:
                best_score = confidence
                best_idx = i

        return self.structure_paths[best_idx]


@dataclass
class EvaluationResult:
    """Result from structure evaluation."""

    structure_path: Path
    reference_path: Optional[Path] = None
    lddt: Optional[float] = None
    lddt_per_residue: Optional[list[float]] = None
    tm_score: Optional[float] = None
    qs_score: Optional[float] = None
    rmsd: Optional[float] = None
    gdt_ts: Optional[float] = None
    gdt_ha: Optional[float] = None
    clash_score: Optional[float] = None
    clash_count: Optional[int] = None
    contact_energy: Optional[float] = None
    contact_energy_per_residue: Optional[float] = None
    rosetta_total_score: Optional[float] = None
    sasa_total: Optional[float] = None
    interface_bsa_total: Optional[float] = None
    salt_bridge_count: Optional[int] = None
    salt_bridge_count_interchain: Optional[int] = None
    openmm_potential_energy_kj_mol: Optional[float] = None
    openmm_gbsa_energy_kj_mol: Optional[float] = None
    rosetta_score_jd2_total_score: Optional[float] = None
    rosetta_cartesian_ddg: Optional[float] = None
    foldx_ddg_kcal_mol: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "structure_path": str(self.structure_path),
            "reference_path": str(self.reference_path) if self.reference_path else None,
            "lddt": self.lddt,
            "tm_score": self.tm_score,
            "qs_score": self.qs_score,
            "rmsd": self.rmsd,
            "gdt_ts": self.gdt_ts,
            "gdt_ha": self.gdt_ha,
            "clash_score": self.clash_score,
            "clash_count": self.clash_count,
            "contact_energy": self.contact_energy,
            "contact_energy_per_residue": self.contact_energy_per_residue,
            "rosetta_total_score": self.rosetta_total_score,
            "sasa_total": self.sasa_total,
            "interface_bsa_total": self.interface_bsa_total,
            "salt_bridge_count": self.salt_bridge_count,
            "salt_bridge_count_interchain": self.salt_bridge_count_interchain,
            "openmm_potential_energy_kj_mol": self.openmm_potential_energy_kj_mol,
            "openmm_gbsa_energy_kj_mol": self.openmm_gbsa_energy_kj_mol,
            "rosetta_score_jd2_total_score": self.rosetta_score_jd2_total_score,
            "rosetta_cartesian_ddg": self.rosetta_cartesian_ddg,
            "foldx_ddg_kcal_mol": self.foldx_ddg_kcal_mol,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result from comparing multiple predictors."""

    job_id: str
    prediction_results: dict[str, PredictionResult]
    evaluation_results: dict[str, EvaluationResult]
    best_predictor: Optional[str] = None
    ranking: list[tuple[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InstallationStatus:
    """Status of a tool installation."""

    name: str
    installed: bool
    version: Optional[str] = None
    latest_version: Optional[str] = None
    path: Optional[Path] = None
    gpu_available: bool = False
    error_message: Optional[str] = None

    @property
    def needs_update(self) -> bool:
        if not self.installed or not self.version or not self.latest_version:
            return False
        return self.version != self.latest_version
