"""OpenMM GBSA energy metric (implicit solvent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric


class OpenMMGBSAMetric(BaseMetric):
    """
    Compute OpenMM potential energy with an implicit solvent model (OBC2) and report
    total potential and GBSA contribution (if identifiable).

    Notes:
    - Protein-only, PDB input is the primary target.
    - Ligands/modified residues generally require parameterization and are not handled here.
    """

    name = "openmm_gbsa"
    description = "OpenMM implicit-solvent energy (OBC2) and GBSA term"
    requires_reference = False

    def __init__(self, minimize: bool = True, max_iters: int = 200):
        self.minimize = bool(minimize)
        self.max_iters = int(max_iters)

    def is_available(self) -> bool:
        try:
            import openmm  # noqa: F401
            import openmm.app  # noqa: F401

            return True
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "OpenMM (pip install openmm) + a compatible amber forcefield XML"

    def compute(
        self, model_path: Path, reference_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise EvaluationError(self.name, "OpenMM not available")

        import openmm
        from openmm import unit
        from openmm import LocalEnergyMinimizer
        from openmm.app import ForceField, Modeller, PDBFile, NoCutoff, HBonds, Simulation, OBC2

        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")

        pdb = PDBFile(str(model_path))
        modeller = Modeller(pdb.topology, pdb.positions)

        # Add hydrogens; this will fail for unknown residues.
        try:
            ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
            modeller.addHydrogens(ff)
        except Exception as e:
            raise EvaluationError(
                self.name,
                "Failed to parameterize structure. This metric currently supports standard protein residues only.",
                original_error=e,
            )

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
            implicitSolvent=OBC2,
        )

        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = None
        try:
            from openmm import Platform

            platform = Platform.getPlatformByName("CUDA")
        except Exception:
            platform = None

        try:
            if platform is None:
                sim = Simulation(modeller.topology, system, integrator)
            else:
                sim = Simulation(modeller.topology, system, integrator, platform)
        except Exception as e:
            raise EvaluationError(self.name, "Failed to create OpenMM simulation", original_error=e)

        sim.context.setPositions(modeller.positions)

        if self.minimize:
            LocalEnergyMinimizer.minimize(sim.context, maxIterations=self.max_iters)

        # Group forces to estimate GB contribution.
        gb_group = 1
        other_group = 2
        for f in system.getForces():
            fname = f.__class__.__name__.lower()
            if "gb" in fname or "obc" in fname:
                f.setForceGroup(gb_group)
            else:
                f.setForceGroup(other_group)

        state_total = sim.context.getState(getEnergy=True)
        total = state_total.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        state_gb = sim.context.getState(getEnergy=True, groups=1 << gb_group)
        gb = state_gb.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        return {
            "openmm_potential_energy_kj_mol": float(total),
            "openmm_gbsa_energy_kj_mol": float(gb),
            "minimized": bool(self.minimize),
        }
