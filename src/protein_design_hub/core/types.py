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
    ESM3 = "esm3"


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
    SEQUENCE_RECOVERY = "sequence_recovery"
    DISORDER = "disorder"
    SHAPE_COMPLEMENTARITY = "shape_complementarity"
    # New metrics
    RAMACHANDRAN = "ramachandran"
    ROTAMER_QUALITY = "rotamer_quality"
    MOLPROBITY = "molprobity"
    HBOND_ENERGY = "hbond_energy"
    AGGREGATION_SCORE = "aggregation_score"
    SOLUBILITY = "solubility"


class SecondaryStructure(str, Enum):
    """Secondary structure types."""

    HELIX = "H"
    SHEET = "E"
    COIL = "C"
    ANY = "X"


class ConstraintType(str, Enum):
    """Types of design constraints."""

    DISTANCE = "distance"
    CONTACT = "contact"
    SECONDARY_STRUCTURE = "secondary_structure"
    FIXED_RESIDUE = "fixed_residue"
    ALLOWED_AA = "allowed_aa"
    FORBIDDEN_AA = "forbidden_aa"


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
class DesignConstraint:
    """Design constraint for sequence design."""

    constraint_type: ConstraintType
    positions: list[int] = field(default_factory=list)
    chain: str = "A"
    # For fixed/allowed/forbidden AA
    amino_acids: list[str] = field(default_factory=list)
    # For secondary structure
    secondary_structure: Optional[str] = None
    # For distance constraints
    target_distance: Optional[float] = None
    tolerance: float = 2.0


@dataclass
class ChainInfo:
    """Information about a protein chain."""

    chain_id: str
    sequence: str
    start_residue: int = 1
    molecule_type: MoleculeType = MoleculeType.PROTEIN
    description: str = ""

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass
class MultiChainComplex:
    """A multi-chain protein complex."""

    chains: list[ChainInfo]
    name: str = ""
    ligands: list[str] = field(default_factory=list)  # SMILES strings

    @property
    def total_residues(self) -> int:
        return sum(len(c) for c in self.chains)

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    @property
    def chain_ids(self) -> list[str]:
        return [c.chain_id for c in self.chains]

    def get_chain(self, chain_id: str) -> Optional[ChainInfo]:
        for c in self.chains:
            if c.chain_id == chain_id:
                return c
        return None

    @classmethod
    def from_fasta(cls, fasta_str: str, name: str = "") -> "MultiChainComplex":
        """Parse multi-chain FASTA with : separator or multiple entries."""
        chains = []
        lines = fasta_str.strip().split("\n")
        current_header = ""
        current_seq = []
        chain_idx = 0

        for line in lines:
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq)
                    # Check for chain separator
                    if ":" in seq:
                        for i, part in enumerate(seq.split(":")):
                            if part:
                                chain_id = chr(ord("A") + chain_idx)
                                chains.append(ChainInfo(
                                    chain_id=chain_id,
                                    sequence=part,
                                    description=current_header
                                ))
                                chain_idx += 1
                    else:
                        chain_id = chr(ord("A") + chain_idx)
                        chains.append(ChainInfo(
                            chain_id=chain_id,
                            sequence=seq,
                            description=current_header
                        ))
                        chain_idx += 1
                    current_seq = []
                current_header = line[1:].strip()
            else:
                current_seq.append(line.strip())

        # Last sequence
        if current_seq:
            seq = "".join(current_seq)
            if ":" in seq:
                for i, part in enumerate(seq.split(":")):
                    if part:
                        chain_id = chr(ord("A") + chain_idx)
                        chains.append(ChainInfo(
                            chain_id=chain_id,
                            sequence=part,
                            description=current_header
                        ))
                        chain_idx += 1
            else:
                chain_id = chr(ord("A") + chain_idx)
                chains.append(ChainInfo(
                    chain_id=chain_id,
                    sequence=seq,
                    description=current_header
                ))

        return cls(chains=chains, name=name)

    def to_fasta(self, use_separator: bool = False) -> str:
        """Convert to FASTA format."""
        if use_separator and len(self.chains) > 1:
            seqs = ":".join(c.sequence for c in self.chains)
            return f">{self.name or 'complex'}\n{seqs}"
        else:
            lines = []
            for c in self.chains:
                header = c.description or f"{self.name}_{c.chain_id}"
                lines.append(f">{header}")
                lines.append(c.sequence)
            return "\n".join(lines)


@dataclass
class PerChainMetrics:
    """Metrics computed per chain."""

    chain_id: str
    plddt: Optional[float] = None
    plddt_per_residue: Optional[list[float]] = None
    tm_score: Optional[float] = None
    rmsd: Optional[float] = None
    sequence_length: int = 0
    secondary_structure: Optional[str] = None
    disorder_fraction: Optional[float] = None


@dataclass
class MolProbityResult:
    """MolProbity validation results."""

    ramachandran_favored: float = 0.0
    ramachandran_allowed: float = 0.0
    ramachandran_outliers: float = 0.0
    rotamer_outliers: float = 0.0
    cbeta_deviations: int = 0
    clashscore: float = 0.0
    molprobity_score: float = 0.0
    outlier_residues: list[dict] = field(default_factory=list)


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
    cad_score: Optional[float] = None
    cad_score_per_residue: Optional[list[float]] = None
    voromqa_score: Optional[float] = None
    voromqa_per_residue: Optional[list[float]] = None
    voromqa_residue_count: Optional[int] = None
    voromqa_atom_count: Optional[int] = None
    # Sequence recovery metrics
    sequence_recovery: Optional[float] = None
    sequence_recovery_per_residue: Optional[list[float]] = None
    # Disorder metrics
    disorder_fraction: Optional[float] = None
    disorder_per_residue: Optional[list[float]] = None
    disorder_regions: Optional[list[dict]] = None
    # Shape complementarity metrics
    shape_complementarity: Optional[float] = None
    interface_residues_a: Optional[int] = None
    interface_residues_b: Optional[int] = None
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
            "cad_score": self.cad_score,
            "cad_score_per_residue": self.cad_score_per_residue,
            "voromqa_score": self.voromqa_score,
            "voromqa_per_residue": self.voromqa_per_residue,
            "voromqa_residue_count": self.voromqa_residue_count,
            "voromqa_atom_count": self.voromqa_atom_count,
            "sequence_recovery": self.sequence_recovery,
            "disorder_fraction": self.disorder_fraction,
            "shape_complementarity": self.shape_complementarity,
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
