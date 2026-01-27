"""OpenStructure runner for executing OST commands in micromamba environment."""

import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class OpenStructureRunner:
    """
    Run OpenStructure commands in the micromamba 'ost' environment.

    OpenStructure is installed in a separate micromamba environment.
    This runner executes Python code within that environment.
    """

    OST_ENV_NAME = "ost"

    def __init__(self):
        self._micromamba_path: Optional[Path] = None
        self._ost_available: Optional[bool] = None

    @property
    def micromamba_path(self) -> Optional[Path]:
        """Get path to micromamba executable."""
        if self._micromamba_path is not None:
            return self._micromamba_path

        # Check common locations
        locations = [
            Path.home() / "bin" / "micromamba",
            Path.home() / ".local" / "bin" / "micromamba",
            Path("/usr/local/bin/micromamba"),
            Path("/usr/bin/micromamba"),
        ]

        for loc in locations:
            if loc.exists():
                self._micromamba_path = loc
                return loc

        # Try which
        result = shutil.which("micromamba")
        if result:
            self._micromamba_path = Path(result)
            return self._micromamba_path

        return None

    def is_available(self) -> bool:
        """Check if OpenStructure is available in micromamba environment."""
        if self._ost_available is not None:
            return self._ost_available

        if not self.micromamba_path:
            self._ost_available = False
            return False

        # Check if ost environment exists and has openstructure
        try:
            result = subprocess.run(
                [str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
                 "python", "-c", "import ost; print(ost.__version__)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._ost_available = result.returncode == 0
            return self._ost_available
        except Exception:
            self._ost_available = False
            return False

    def get_version(self) -> Optional[str]:
        """Get OpenStructure version."""
        if not self.is_available():
            return None

        try:
            result = subprocess.run(
                [str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
                 "python", "-c", "import ost; print(ost.__version__)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def run_script(self, script: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Run a Python script in the OST environment.

        Args:
            script: Python code to execute.
            timeout: Timeout in seconds.

        Returns:
            Dictionary with stdout, stderr, and returncode.
        """
        if not self.is_available():
            raise RuntimeError("OpenStructure not available in micromamba environment")

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
                 "python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        finally:
            Path(script_path).unlink(missing_ok=True)

    def compute_lddt(
        self,
        model_path: Path,
        reference_path: Path,
        inclusion_radius: float = 15.0,
        sequence_separation: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute lDDT score using OpenStructure.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            inclusion_radius: Radius for including atoms.
            sequence_separation: Minimum sequence separation.

        Returns:
            Dictionary with lDDT scores.
        """
        script = f'''
import json
import sys
from ost.io import LoadPDB, LoadMMCIF
from ost.mol.alg import lDDTScorer, lDDTSettings
from ost.mol import EntityViewList

def load_structure(path):
    if path.endswith(('.cif', '.mmcif')):
        return LoadMMCIF(path)
    return LoadPDB(path)

try:
    model = load_structure("{model_path}")
    reference = load_structure("{reference_path}")

    # Select protein backbone atoms
    model_sel = model.Select("peptide=true and aname=CA,C,N,O")
    ref_sel = reference.Select("peptide=true and aname=CA,C,N,O")

    if model_sel.GetAtomCount() == 0:
        print(json.dumps({{"error": "No protein atoms in model"}}))
        sys.exit(1)
    if ref_sel.GetAtomCount() == 0:
        print(json.dumps({{"error": "No protein atoms in reference"}}))
        sys.exit(1)

    # Create settings with standard lDDT thresholds
    settings = lDDTSettings(
        {inclusion_radius},  # inclusion_radius
        {sequence_separation},  # sequence_separation
        [0.5, 1.0, 2.0, 4.0],  # distance thresholds
        "lddt"  # label
    )

    # Create EntityViewList for references
    ref_list = EntityViewList()
    ref_list.append(ref_sel)

    # Create scorer and get global lDDT
    scorer = lDDTScorer(ref_list, model_sel, settings)
    global_lddt = float(scorer.global_score)

    # Get per-residue scores (convert to plain floats)
    per_residue = []
    if scorer.local_scores:
        for score in scorer.local_scores:
            # lDDTLocalScore has a local_lddt attribute
            if hasattr(score, 'local_lddt'):
                per_residue.append(float(score.local_lddt))
            elif hasattr(score, 'score'):
                per_residue.append(float(score.score))
            else:
                try:
                    per_residue.append(float(score))
                except:
                    pass

    result = {{
        "lddt": global_lddt,
        "lddt_per_residue": per_residue,
        "num_residues": len(per_residue),
        "total_contacts": int(scorer.total_contacts),
        "conserved_contacts": int(scorer.conserved_contacts),
    }}
    print(json.dumps(result))

except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
    sys.exit(1)
'''

        result = self.run_script(script)

        if result["returncode"] != 0:
            error_msg = result["stderr"] or result["stdout"]
            raise RuntimeError(f"lDDT computation failed: {error_msg}")

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse lDDT result: {result['stdout']}")

    def compute_qs_score(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """
        Compute QS-score using OpenStructure.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with QS-score values.
        """
        script = f'''
import json
import sys
from ost.io import LoadPDB, LoadMMCIF
from ost.mol.alg import qsscore

def load_structure(path):
    if path.endswith(('.cif', '.mmcif')):
        return LoadMMCIF(path)
    return LoadPDB(path)

try:
    model = load_structure("{model_path}")
    reference = load_structure("{reference_path}")

    model_chains = [ch.GetName() for ch in model.chains]
    ref_chains = [ch.GetName() for ch in reference.chains]

    # Compute QS-score
    qs_scorer = qsscore.QSScorer(
        reference.Select("peptide=true"),
        model.Select("peptide=true"),
    )

    result = {{
        "qs_score": qs_scorer.global_score,
        "qs_best": qs_scorer.best_score if hasattr(qs_scorer, "best_score") else qs_scorer.global_score,
        "model_chains": model_chains,
        "reference_chains": ref_chains,
    }}
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
'''

        result = self.run_script(script)

        if result["returncode"] != 0:
            error_msg = result["stderr"] or result["stdout"]
            raise RuntimeError(f"QS-score computation failed: {error_msg}")

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse QS-score result: {result['stdout']}")

    def compute_rmsd(
        self,
        model_path: Path,
        reference_path: Path,
        atoms: str = "CA",
    ) -> Dict[str, Any]:
        """
        Compute RMSD using OpenStructure.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            atoms: Atom selection (CA, backbone, heavy, all).

        Returns:
            Dictionary with RMSD values.
        """
        # Build atom selection
        if atoms == "CA":
            selection = "peptide=true and aname=CA"
        elif atoms == "backbone":
            selection = "peptide=true and aname=CA,C,N,O"
        elif atoms == "heavy":
            selection = "peptide=true and ele!=H"
        else:
            selection = "peptide=true"

        script = f'''
import json
import sys
from ost.io import LoadPDB, LoadMMCIF
from ost.mol.alg import Superpose

def load_structure(path):
    if path.endswith(('.cif', '.mmcif')):
        return LoadMMCIF(path)
    return LoadPDB(path)

try:
    model = load_structure("{model_path}")
    reference = load_structure("{reference_path}")

    model_sel = model.Select("{selection}")
    ref_sel = reference.Select("{selection}")

    if model_sel.GetAtomCount() == 0:
        print(json.dumps({{"error": "No atoms in model selection"}}))
        sys.exit(1)
    if ref_sel.GetAtomCount() == 0:
        print(json.dumps({{"error": "No atoms in reference selection"}}))
        sys.exit(1)

    # Superpose
    result = Superpose(model_sel, ref_sel)

    output = {{
        "rmsd": result.rmsd,
        "num_atoms": model_sel.GetAtomCount(),
    }}
    print(json.dumps(output))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
'''

        result = self.run_script(script)

        if result["returncode"] != 0:
            error_msg = result["stderr"] or result["stdout"]
            raise RuntimeError(f"RMSD computation failed: {error_msg}")

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse RMSD result: {result['stdout']}")

    def load_structure_info(self, structure_path: Path) -> Dict[str, Any]:
        """
        Load structure and get basic information.

        Args:
            structure_path: Path to structure file.

        Returns:
            Dictionary with structure information.
        """
        script = f'''
import json
import sys
from ost.io import LoadPDB, LoadMMCIF

def load_structure(path):
    if path.endswith(('.cif', '.mmcif')):
        return LoadMMCIF(path)
    return LoadPDB(path)

try:
    structure = load_structure("{structure_path}")

    chains = []
    for chain in structure.chains:
        chains.append({{
            "name": chain.GetName(),
            "num_residues": chain.GetResidueCount(),
            "num_atoms": chain.GetAtomCount(),
        }})

    result = {{
        "num_chains": len(chains),
        "num_residues": structure.GetResidueCount(),
        "num_atoms": structure.GetAtomCount(),
        "chains": chains,
    }}
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
'''

        result = self.run_script(script, timeout=60)

        if result["returncode"] != 0:
            error_msg = result["stderr"] or result["stdout"]
            raise RuntimeError(f"Structure loading failed: {error_msg}")

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse structure info: {result['stdout']}")

    def run_compare_structures(
        self,
        model_path: Path,
        reference_path: Path,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run OST compare-structures CLI action for comprehensive evaluation.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            metrics: List of metrics to compute. If None, computes all.

        Returns:
            Dictionary with all computed metrics.
        """
        if not self.is_available():
            raise RuntimeError("OpenStructure not available in micromamba environment")

        # Build command with all metric flags
        cmd = [
            str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
            "ost", "compare-structures",
            "-m", str(model_path),
            "-r", str(reference_path),
            "-o", "-",  # Output to stdout
            "-of", "json",
        ]

        # Add metric flags
        if metrics is None:
            metrics = [
                "lddt", "bb-lddt", "ilddt",
                "qs-score", "dockq", "ics", "ips", "patch-scores",
                "rigid-scores", "local-lddt",
            ]

        for metric in metrics:
            cmd.append(f"--{metric}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                # Try to parse partial results or return error
                if result.stdout:
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        pass
                raise RuntimeError(f"compare-structures failed: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise RuntimeError("compare-structures timed out after 10 minutes")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse compare-structures output: {e}")

    def compute_all_metrics(
        self,
        model_path: Path,
        reference_path: Path,
        use_cli: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute all OpenStructure metrics at global, per-residue, and interface levels.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            use_cli: If True, use OST CLI for comprehensive metrics.

        Returns:
            Comprehensive dictionary with all metrics at all levels.
        """
        # Try CLI-based approach first for comprehensive metrics
        if use_cli:
            try:
                cli_result = self.run_compare_structures(model_path, reference_path)
                return self._transform_cli_result(cli_result)
            except Exception as e:
                # Fall back to script-based approach
                pass

        script = f'''
import json
import sys
from ost.io import LoadPDB, LoadMMCIF
from ost.mol.alg import lDDTScorer, lDDTSettings, Superpose
from ost.mol.alg import qsscore
from ost.mol import EntityViewList

def load_structure(path):
    if path.endswith(('.cif', '.mmcif')):
        return LoadMMCIF(path)
    return LoadPDB(path)

try:
    model = load_structure("{model_path}")
    reference = load_structure("{reference_path}")

    results = {{
        "global": {{}},
        "per_residue": {{}},
        "per_chain": {{}},
        "interface": {{}},
    }}

    # ========== lDDT Metrics ==========
    model_bb = model.Select("peptide=true and aname=CA,C,N,O")
    ref_bb = reference.Select("peptide=true and aname=CA,C,N,O")

    if model_bb.GetAtomCount() > 0 and ref_bb.GetAtomCount() > 0:
        # Global lDDT
        settings = lDDTSettings(15.0, 0, [0.5, 1.0, 2.0, 4.0], "lddt")
        ref_list = EntityViewList()
        ref_list.append(ref_bb)
        scorer = lDDTScorer(ref_list, model_bb, settings)

        results["global"]["lddt"] = float(scorer.global_score)
        results["global"]["lddt_total_contacts"] = int(scorer.total_contacts)
        results["global"]["lddt_conserved_contacts"] = int(scorer.conserved_contacts)

        # Per-residue lDDT
        per_res_lddt = []
        residue_info = []
        if scorer.local_scores:
            for score in scorer.local_scores:
                per_res_lddt.append(float(score.local_lddt))
                residue_info.append({{
                    "chain": score.cname,
                    "residue_name": score.rname,
                    "residue_num": score.rnum,
                    "lddt": float(score.local_lddt),
                    "conserved_dist": int(score.conserved_dist),
                    "total_dist": int(score.total_dist),
                    "is_assessed": score.is_assessed,
                }})

        results["per_residue"]["lddt"] = per_res_lddt
        results["per_residue"]["lddt_details"] = residue_info

        # Per-chain lDDT
        chain_lddt = {{}}
        for info in residue_info:
            chain = info["chain"]
            if chain not in chain_lddt:
                chain_lddt[chain] = {{"scores": [], "conserved": 0, "total": 0}}
            chain_lddt[chain]["scores"].append(info["lddt"])
            chain_lddt[chain]["conserved"] += info["conserved_dist"]
            chain_lddt[chain]["total"] += info["total_dist"]

        for chain, data in chain_lddt.items():
            if data["scores"]:
                data["mean_lddt"] = sum(data["scores"]) / len(data["scores"])
                data["num_residues"] = len(data["scores"])
            del data["scores"]

        results["per_chain"]["lddt"] = chain_lddt

    # ========== RMSD Metrics ==========
    model_ca = model.Select("peptide=true and aname=CA")
    ref_ca = reference.Select("peptide=true and aname=CA")

    if model_ca.GetAtomCount() > 0 and ref_ca.GetAtomCount() > 0:
        try:
            sup_result = Superpose(model_ca, ref_ca)
            results["global"]["rmsd_ca"] = float(sup_result.rmsd)
            results["global"]["rmsd_num_atoms"] = model_ca.GetAtomCount()
        except Exception as e:
            results["global"]["rmsd_error"] = str(e)

    # Backbone RMSD
    if model_bb.GetAtomCount() > 0 and ref_bb.GetAtomCount() > 0:
        try:
            sup_result_bb = Superpose(model_bb, ref_bb)
            results["global"]["rmsd_backbone"] = float(sup_result_bb.rmsd)
        except Exception as e:
            results["global"]["rmsd_backbone_error"] = str(e)

    # ========== QS-score Metrics (Interface) ==========
    model_pep = model.Select("peptide=true")
    ref_pep = reference.Select("peptide=true")

    model_chains = [ch.GetName() for ch in model.chains if ch.GetResidueCount() > 0]
    ref_chains = [ch.GetName() for ch in reference.chains if ch.GetResidueCount() > 0]

    results["global"]["model_chains"] = model_chains
    results["global"]["reference_chains"] = ref_chains

    # Only compute QS-score for multimeric structures
    if len(model_chains) > 1 and len(ref_chains) > 1:
        try:
            qs_scorer = qsscore.QSScorer(ref_pep, model_pep)
            results["global"]["qs_score"] = float(qs_scorer.global_score)
            if hasattr(qs_scorer, "best_score"):
                results["global"]["qs_best"] = float(qs_scorer.best_score)

            # Interface-level metrics
            if hasattr(qs_scorer, "chain_mapping"):
                results["interface"]["chain_mapping"] = dict(qs_scorer.chain_mapping)

            # Get interface residue information if available
            if hasattr(qs_scorer, "mapped_target") and hasattr(qs_scorer, "mapped_model"):
                results["interface"]["mapped_target_chains"] = [ch.GetName() for ch in qs_scorer.mapped_target.chains]
                results["interface"]["mapped_model_chains"] = [ch.GetName() for ch in qs_scorer.mapped_model.chains]

        except Exception as e:
            results["interface"]["qs_error"] = str(e)
    else:
        results["interface"]["qs_note"] = "QS-score requires multimeric structures (>1 chain)"

    # ========== Chain-level statistics ==========
    for chain in model.chains:
        cname = chain.GetName()
        if cname not in results["per_chain"]:
            results["per_chain"][cname] = {{}}

        results["per_chain"][cname]["num_residues"] = chain.GetResidueCount()
        results["per_chain"][cname]["num_atoms"] = chain.GetAtomCount()

    # ========== Summary statistics ==========
    if results["per_residue"].get("lddt"):
        lddt_vals = results["per_residue"]["lddt"]
        results["global"]["lddt_mean"] = sum(lddt_vals) / len(lddt_vals) if lddt_vals else 0
        results["global"]["lddt_min"] = min(lddt_vals) if lddt_vals else 0
        results["global"]["lddt_max"] = max(lddt_vals) if lddt_vals else 0

        # Categorize residues by lDDT quality
        very_low = sum(1 for x in lddt_vals if x < 0.5)
        low = sum(1 for x in lddt_vals if 0.5 <= x < 0.7)
        confident = sum(1 for x in lddt_vals if 0.7 <= x < 0.9)
        very_high = sum(1 for x in lddt_vals if x >= 0.9)

        results["global"]["lddt_quality_categories"] = {{
            "very_low_lt_50": very_low,
            "low_50_70": low,
            "confident_70_90": confident,
            "very_high_gt_90": very_high,
        }}

    print(json.dumps(results))

except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
    sys.exit(1)
'''

        result = self.run_script(script, timeout=300)

        if result["returncode"] != 0:
            error_msg = result["stderr"] or result["stdout"]
            raise RuntimeError(f"Metrics computation failed: {error_msg}")

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse metrics result: {result['stdout']}")

    def _transform_cli_result(self, cli_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform OST CLI compare-structures output to our standard format.

        Args:
            cli_result: Raw output from ost compare-structures.

        Returns:
            Transformed dictionary with global, per_residue, per_chain, and interface sections.
        """
        results = {
            "global": {},
            "per_residue": {},
            "per_chain": {},
            "interface": {},
        }

        # Extract global lDDT
        if "lddt" in cli_result:
            results["global"]["lddt"] = cli_result["lddt"]
        if "bb_lddt" in cli_result:
            results["global"]["bb_lddt"] = cli_result["bb_lddt"]
        if "ilddt" in cli_result:
            results["global"]["ilddt"] = cli_result["ilddt"]

        # Extract QS-score
        if "qs_global" in cli_result:
            results["global"]["qs_score"] = cli_result["qs_global"]
        if "qs_best" in cli_result:
            results["global"]["qs_best"] = cli_result["qs_best"]

        # Extract rigid scores (GDT, RMSD)
        if "gdtts" in cli_result:
            results["global"]["gdt_ts"] = cli_result["gdtts"]
        if "gdtha" in cli_result:
            results["global"]["gdt_ha"] = cli_result["gdtha"]
        if "rmsd" in cli_result:
            results["global"]["rmsd_ca"] = cli_result["rmsd"]

        # Extract CAD score
        if "cad_score" in cli_result:
            results["global"]["cad_score"] = cli_result["cad_score"]

        # ========== DockQ and Interface Metrics ==========
        if "dockq_scores" in cli_result:
            dockq_data = cli_result["dockq_scores"]
            if isinstance(dockq_data, list) and len(dockq_data) > 0:
                # Average DockQ across interfaces
                avg_dockq = sum(d.get("dockq", 0) for d in dockq_data) / len(dockq_data)
                results["global"]["dockq"] = avg_dockq
                results["interface"]["dockq_details"] = dockq_data

                # Per-interface breakdown
                for i, dq in enumerate(dockq_data):
                    interface_name = f"interface_{i+1}"
                    results["interface"][interface_name] = {
                        "dockq": dq.get("dockq"),
                        "fnat": dq.get("fnat"),
                        "fnonnat": dq.get("fnonnat"),
                        "irmsd": dq.get("irmsd"),
                        "lrmsd": dq.get("lrmsd"),
                        "chain1_mdl": dq.get("chain1", {}).get("mdl"),
                        "chain1_ref": dq.get("chain1", {}).get("ref"),
                        "chain2_mdl": dq.get("chain2", {}).get("mdl"),
                        "chain2_ref": dq.get("chain2", {}).get("ref"),
                    }
            elif isinstance(dockq_data, dict):
                results["global"]["dockq"] = dockq_data.get("dockq")
                results["interface"]["dockq_details"] = dockq_data

        # ========== ICS (Interface Contact Similarity) ==========
        if "ics_precision" in cli_result:
            results["interface"]["ics_precision"] = cli_result["ics_precision"]
        if "ics_recall" in cli_result:
            results["interface"]["ics_recall"] = cli_result["ics_recall"]
        if "ics" in cli_result:
            results["global"]["ics"] = cli_result["ics"]
            results["interface"]["ics"] = cli_result["ics"]

        # ========== IPS (Interface Patch Similarity) ==========
        if "ips" in cli_result:
            results["global"]["ips"] = cli_result["ips"]
            results["interface"]["ips"] = cli_result["ips"]

        # ========== Patch Scores (CASP15) ==========
        if "patch_scores" in cli_result:
            patch_data = cli_result["patch_scores"]
            if isinstance(patch_data, list) and len(patch_data) > 0:
                avg_patch = sum(p.get("score", 0) for p in patch_data) / len(patch_data)
                results["global"]["patch_score"] = avg_patch
                results["interface"]["patch_scores"] = patch_data
            elif isinstance(patch_data, (int, float)):
                results["global"]["patch_score"] = patch_data

        # ========== Per-residue lDDT ==========
        if "local_lddt" in cli_result:
            local_lddt = cli_result["local_lddt"]
            if isinstance(local_lddt, list):
                lddt_values = []
                lddt_details = []
                for item in local_lddt:
                    if isinstance(item, dict):
                        lddt_values.append(item.get("lddt", 0))
                        lddt_details.append({
                            "chain": item.get("chain"),
                            "residue_name": item.get("resname"),
                            "residue_num": item.get("resnum"),
                            "lddt": item.get("lddt"),
                        })
                    elif isinstance(item, (int, float)):
                        lddt_values.append(item)

                results["per_residue"]["lddt"] = lddt_values
                if lddt_details:
                    results["per_residue"]["lddt_details"] = lddt_details

                # Calculate statistics
                if lddt_values:
                    results["global"]["lddt_mean"] = sum(lddt_values) / len(lddt_values)
                    results["global"]["lddt_min"] = min(lddt_values)
                    results["global"]["lddt_max"] = max(lddt_values)

                    # Quality categories
                    very_low = sum(1 for x in lddt_values if x < 0.5)
                    low = sum(1 for x in lddt_values if 0.5 <= x < 0.7)
                    confident = sum(1 for x in lddt_values if 0.7 <= x < 0.9)
                    very_high = sum(1 for x in lddt_values if x >= 0.9)

                    results["global"]["lddt_quality_categories"] = {
                        "very_low_lt_50": very_low,
                        "low_50_70": low,
                        "confident_70_90": confident,
                        "very_high_gt_90": very_high,
                    }

        # ========== Chain mapping ==========
        if "chain_mapping" in cli_result:
            results["interface"]["chain_mapping"] = cli_result["chain_mapping"]

        # ========== Alignment info ==========
        if "aln" in cli_result:
            results["interface"]["alignment"] = cli_result["aln"]

        # ========== Per-chain metrics ==========
        if "per_chain_lddt" in cli_result:
            for chain_data in cli_result["per_chain_lddt"]:
                chain_name = chain_data.get("chain", "unknown")
                results["per_chain"][chain_name] = {
                    "lddt": chain_data.get("lddt"),
                    "num_residues": chain_data.get("num_residues"),
                }

        return results

    def compute_dockq(
        self,
        model_path: Path,
        reference_path: Path,
        capri_peptide: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute DockQ score for interface quality assessment.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            capri_peptide: Use CAPRI peptide mode for small peptides.

        Returns:
            Dictionary with DockQ scores and components.
        """
        cmd = [
            str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
            "ost", "compare-structures",
            "-m", str(model_path),
            "-r", str(reference_path),
            "-o", "-",
            "-of", "json",
            "--dockq",
        ]

        if capri_peptide:
            cmd.append("--dockq-capri-peptide")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(f"DockQ computation failed: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise RuntimeError("DockQ computation timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse DockQ output: {e}")

    def compute_patch_scores(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """
        Compute Patch Scores (CASP15 local interface quality).

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with patch scores.
        """
        cmd = [
            str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
            "ost", "compare-structures",
            "-m", str(model_path),
            "-r", str(reference_path),
            "-o", "-",
            "-of", "json",
            "--patch-scores",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Patch scores computation failed: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise RuntimeError("Patch scores computation timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Patch scores output: {e}")

    def compute_ics_ips(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """
        Compute ICS (Interface Contact Similarity) and IPS (Interface Patch Similarity).

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with ICS and IPS metrics.
        """
        cmd = [
            str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
            "ost", "compare-structures",
            "-m", str(model_path),
            "-r", str(reference_path),
            "-o", "-",
            "-of", "json",
            "--ics",
            "--ips",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ICS/IPS computation failed: {result.stderr}")

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise RuntimeError("ICS/IPS computation timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ICS/IPS output: {e}")


    def compute_lddt_pli(
        self,
        model_path: Path,
        reference_path: Path,
        model_ligands: Optional[List[Path]] = None,
        reference_ligands: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """
        Compute LDDT-PLI (Protein-Ligand Interface) scores using OpenStructure.

        This evaluates how well the protein-ligand interface is predicted,
        including both the protein binding site and ligand positioning.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            model_ligands: Optional list of model ligand files (SDF/MOL2).
            reference_ligands: Optional list of reference ligand files.

        Returns:
            Dictionary with LDDT-PLI scores and components.
        """
        cmd = [
            str(self.micromamba_path), "run", "-n", self.OST_ENV_NAME,
            "ost", "compare-ligand-structures",
            "-m", str(model_path),
            "-r", str(reference_path),
            "-o", "-",
            "-of", "json",
            "--lddt-pli",
            "--rmsd",
        ]

        # Add ligand files if provided
        if model_ligands:
            for lig in model_ligands:
                cmd.extend(["-ml", str(lig)])
        if reference_ligands:
            for lig in reference_ligands:
                cmd.extend(["-rl", str(lig)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                # Check if it's just a "no ligands found" situation
                if "no ligand" in result.stderr.lower():
                    return {"error": "No ligands found in structures", "lddt_pli": None}
                raise RuntimeError(f"LDDT-PLI computation failed: {result.stderr}")

            raw_result = json.loads(result.stdout)
            return self._transform_lddt_pli_result(raw_result)

        except subprocess.TimeoutExpired:
            raise RuntimeError("LDDT-PLI computation timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LDDT-PLI output: {e}")

    def _transform_lddt_pli_result(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw LDDT-PLI output to structured format."""
        result = {
            "lddt_pli": None,
            "lddt_lp": None,
            "rmsd": None,
            "ligand_scores": [],
            "binding_site_quality": None,
        }

        # Extract global scores
        if "lddt_pli" in raw:
            result["lddt_pli"] = raw["lddt_pli"]
        if "lddt_lp" in raw:
            result["lddt_lp"] = raw["lddt_lp"]

        # Extract per-ligand scores
        if "ligand_scores" in raw:
            for lig in raw["ligand_scores"]:
                result["ligand_scores"].append({
                    "ligand_id": lig.get("ligand_id"),
                    "lddt_pli": lig.get("lddt_pli"),
                    "rmsd": lig.get("rmsd"),
                    "coverage": lig.get("coverage"),
                    "model_ligand": lig.get("model_ligand"),
                    "reference_ligand": lig.get("reference_ligand"),
                })

        # Extract BiSyRMSD if available
        if "bisyrmsd" in raw:
            result["bisyrmsd"] = raw["bisyrmsd"]

        return result

    def compute_binding_site_similarity(
        self,
        model_path: Path,
        reference_path: Path,
        ligand_center: Optional[tuple] = None,
        radius: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Compute binding site similarity between model and reference.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.
            ligand_center: (x, y, z) center of binding site. If None, auto-detect.
            radius: Radius around center to define binding site.

        Returns:
            Dictionary with binding site metrics.
        """
        script = f'''
import json
import sys
from ost.io import LoadPDB, LoadMMCIF
from ost.mol.alg import lDDTScorer, lDDTSettings, Superpose
from ost.mol import EntityViewList
from ost.geom import Vec3

def load_structure(path):
    if path.endswith(('.cif', '.mmcif')):
        return LoadMMCIF(path)
    return LoadPDB(path)

try:
    model = load_structure("{model_path}")
    reference = load_structure("{reference_path}")

    radius = {radius}

    # Find binding site center
    # If ligands present, use their center
    # Otherwise, use the center of the structure
    center = None

    # Try to find HETATM records (ligands)
    ligand_atoms = reference.Select("ishetatm=true and ele!=H")
    if ligand_atoms.GetAtomCount() > 0:
        # Calculate center of mass of ligands
        coords = [a.GetPos() for a in ligand_atoms.atoms]
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        center = Vec3(cx, cy, cz)

    if center is None:
        print(json.dumps({{"error": "No ligands found to define binding site"}}))
        sys.exit(0)

    # Select binding site residues (within radius of center)
    model_bs = model.Select(f"peptide=true and {{radius}} <> [{{center[0]}},{{center[1]}},{{center[2]}}]")
    ref_bs = reference.Select(f"peptide=true and {{radius}} <> [{{center[0]}},{{center[1]}},{{center[2]}}]")

    if model_bs.GetResidueCount() == 0 or ref_bs.GetResidueCount() == 0:
        print(json.dumps({{"error": "No residues in binding site selection"}}))
        sys.exit(0)

    # Compute lDDT for binding site
    model_bb = model_bs.Select("aname=CA,C,N,O")
    ref_bb = ref_bs.Select("aname=CA,C,N,O")

    settings = lDDTSettings(15.0, 0, [0.5, 1.0, 2.0, 4.0], "lddt")
    ref_list = EntityViewList()
    ref_list.append(ref_bb)
    scorer = lDDTScorer(ref_list, model_bb, settings)

    # Compute RMSD
    model_ca = model_bs.Select("aname=CA")
    ref_ca = ref_bs.Select("aname=CA")
    sup = Superpose(model_ca, ref_ca)

    result = {{
        "binding_site_lddt": float(scorer.global_score),
        "binding_site_rmsd": float(sup.rmsd),
        "num_residues": ref_bs.GetResidueCount(),
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "radius": radius,
    }}

    print(json.dumps(result))

except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
    sys.exit(1)
'''

        result = self.run_script(script, timeout=120)

        if result["returncode"] != 0:
            error_msg = result["stderr"] or result["stdout"]
            raise RuntimeError(f"Binding site analysis failed: {error_msg}")

        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse binding site result: {result['stdout']}")


# Global singleton instance
_runner: Optional[OpenStructureRunner] = None


def get_ost_runner() -> OpenStructureRunner:
    """Get the global OpenStructure runner instance."""
    global _runner
    if _runner is None:
        _runner = OpenStructureRunner()
    return _runner
