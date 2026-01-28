"""Shape complementarity metric for interface evaluation."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from protein_design_hub.evaluation.base import BaseMetric


class ShapeComplementarityMetric(BaseMetric):
    """
    Calculate shape complementarity (Sc) score for protein-protein interfaces.

    Based on Lawrence & Colman (1993) shape complementarity algorithm.
    Sc ranges from 0 (no complementarity) to 1 (perfect complementarity).
    """

    def __init__(
        self,
        interface_distance: float = 8.0,
        density: float = 15.0,
    ):
        """
        Initialize shape complementarity metric.

        Args:
            interface_distance: Distance cutoff for interface residues (Angstroms).
            density: Dot density for surface calculation.
        """
        self.interface_distance = interface_distance
        self.density = density

    @property
    def name(self) -> str:
        return "shape_complementarity"

    @property
    def description(self) -> str:
        return "Shape complementarity (Sc) score for interfaces"

    @property
    def requires_reference(self) -> bool:
        return False  # Can compute on single complex

    def is_available(self) -> bool:
        try:
            from Bio.PDB import PDBParser
            import numpy as np
            return True
        except ImportError:
            return False

    def get_requirements(self) -> str:
        return "Requires: pip install biopython numpy"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        chain_a: str = "A",
        chain_b: str = "B",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute shape complementarity.

        Args:
            model_path: Path to complex structure.
            reference_path: Not used.
            chain_a: First chain ID.
            chain_b: Second chain ID.

        Returns:
            Dictionary with Sc score and details.
        """
        try:
            from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
            import numpy as np

            model_path = Path(model_path)

            # Load structure
            if model_path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            structure = parser.get_structure('complex', str(model_path))

            # Get chains
            chains = {}
            for model in structure:
                for chain in model:
                    chains[chain.id] = chain
                break

            if chain_a not in chains or chain_b not in chains:
                # Try to find any two chains
                chain_ids = list(chains.keys())
                if len(chain_ids) >= 2:
                    chain_a, chain_b = chain_ids[0], chain_ids[1]
                else:
                    return {"error": f"Need at least 2 chains, found: {chain_ids}"}

            # Get interface residues
            interface_a, interface_b = self._get_interface_residues(
                chains[chain_a], chains[chain_b]
            )

            if not interface_a or not interface_b:
                return {
                    "error": "No interface found between chains",
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                }

            # Calculate shape complementarity
            sc_score, details = self._calculate_sc(
                interface_a, interface_b
            )

            return {
                "shape_complementarity": sc_score,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "interface_residues_a": len(interface_a),
                "interface_residues_b": len(interface_b),
                "interface_area_a": details.get("area_a", 0),
                "interface_area_b": details.get("area_b", 0),
                "mean_distance": details.get("mean_distance", 0),
            }

        except ImportError as e:
            return {"error": f"Missing dependency: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_interface_residues(
        self, chain_a, chain_b
    ) -> Tuple[List, List]:
        """Get interface residues between two chains."""
        from Bio.PDB import NeighborSearch
        import numpy as np

        # Get all atoms from each chain
        atoms_a = [atom for residue in chain_a for atom in residue if residue.id[0] == ' ']
        atoms_b = [atom for residue in chain_b for atom in residue if residue.id[0] == ' ']

        if not atoms_a or not atoms_b:
            return [], []

        # Find interface using NeighborSearch
        ns = NeighborSearch(atoms_b)

        interface_residues_a = set()
        interface_residues_b = set()

        for atom in atoms_a:
            neighbors = ns.search(atom.get_coord(), self.interface_distance)
            if neighbors:
                interface_residues_a.add(atom.get_parent())
                for neighbor in neighbors:
                    interface_residues_b.add(neighbor.get_parent())

        return list(interface_residues_a), list(interface_residues_b)

    def _calculate_sc(
        self, interface_a: List, interface_b: List
    ) -> Tuple[float, Dict]:
        """
        Calculate shape complementarity score.

        Simplified implementation based on dot product of surface normals.
        """
        import numpy as np

        # Get surface points and normals (simplified)
        points_a, normals_a = self._get_surface_points(interface_a)
        points_b, normals_b = self._get_surface_points(interface_b)

        if len(points_a) == 0 or len(points_b) == 0:
            return 0.0, {}

        # For each point on surface A, find nearest point on surface B
        # and compute normal alignment
        sc_values = []

        for i, (pa, na) in enumerate(zip(points_a, normals_a)):
            # Find nearest point on B
            distances = np.linalg.norm(points_b - pa, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]

            if nearest_dist < self.interface_distance:
                # Compute alignment of normals (should point towards each other)
                nb = normals_b[nearest_idx]
                alignment = -np.dot(na, nb)  # Negative because normals should oppose

                # Weight by distance
                weight = np.exp(-nearest_dist / 3.0)
                sc_values.append(alignment * weight)

        if not sc_values:
            return 0.0, {}

        # Average Sc score
        sc_score = max(0, min(1, (np.mean(sc_values) + 1) / 2))

        # Calculate additional details
        all_distances = []
        for pa in points_a:
            distances = np.linalg.norm(points_b - pa, axis=1)
            all_distances.append(np.min(distances))

        details = {
            "mean_distance": np.mean(all_distances) if all_distances else 0,
            "area_a": len(points_a) * 1.0,  # Approximate area
            "area_b": len(points_b) * 1.0,
        }

        return sc_score, details

    def _get_surface_points(self, residues: List) -> Tuple:
        """Get surface points and normals for residues."""
        import numpy as np

        points = []
        normals = []

        for residue in residues:
            # Use CA position as point
            if 'CA' not in residue:
                continue

            ca = residue['CA'].get_coord()
            points.append(ca)

            # Estimate normal from CB direction (or approximate)
            if 'CB' in residue:
                cb = residue['CB'].get_coord()
                normal = cb - ca
                normal = normal / (np.linalg.norm(normal) + 1e-8)
            else:
                # For glycine, use C-N direction
                if 'C' in residue and 'N' in residue:
                    c = residue['C'].get_coord()
                    n = residue['N'].get_coord()
                    normal = np.cross(c - ca, n - ca)
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                else:
                    normal = np.array([0, 0, 1])

            normals.append(normal)

        return np.array(points), np.array(normals)
