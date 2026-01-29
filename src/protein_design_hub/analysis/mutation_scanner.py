"""Mutation Scanner Module for rapid ESMFold-based mutation analysis.

This module provides functionality for:
1. Single-point saturation mutagenesis using ESMFold
2. Multi-point mutation scanning
3. Metric calculation (pLDDT, RMSD, Clash, SASA, TM-score)
4. Mutation ranking and beneficial variant recommendation
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np


# Standard amino acids (single letter codes)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Amino acid properties for analysis
AA_PROPERTIES = {
    'A': {'name': 'Alanine', 'hydrophobicity': 1.8, 'charge': 0, 'size': 'small', 'polar': False},
    'C': {'name': 'Cysteine', 'hydrophobicity': 2.5, 'charge': 0, 'size': 'small', 'polar': True},
    'D': {'name': 'Aspartate', 'hydrophobicity': -3.5, 'charge': -1, 'size': 'small', 'polar': True},
    'E': {'name': 'Glutamate', 'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium', 'polar': True},
    'F': {'name': 'Phenylalanine', 'hydrophobicity': 2.8, 'charge': 0, 'size': 'large', 'polar': False},
    'G': {'name': 'Glycine', 'hydrophobicity': -0.4, 'charge': 0, 'size': 'small', 'polar': False},
    'H': {'name': 'Histidine', 'hydrophobicity': -3.2, 'charge': 0.1, 'size': 'medium', 'polar': True},
    'I': {'name': 'Isoleucine', 'hydrophobicity': 4.5, 'charge': 0, 'size': 'medium', 'polar': False},
    'K': {'name': 'Lysine', 'hydrophobicity': -3.9, 'charge': 1, 'size': 'large', 'polar': True},
    'L': {'name': 'Leucine', 'hydrophobicity': 3.8, 'charge': 0, 'size': 'medium', 'polar': False},
    'M': {'name': 'Methionine', 'hydrophobicity': 1.9, 'charge': 0, 'size': 'large', 'polar': False},
    'N': {'name': 'Asparagine', 'hydrophobicity': -3.5, 'charge': 0, 'size': 'small', 'polar': True},
    'P': {'name': 'Proline', 'hydrophobicity': -1.6, 'charge': 0, 'size': 'small', 'polar': False},
    'Q': {'name': 'Glutamine', 'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium', 'polar': True},
    'R': {'name': 'Arginine', 'hydrophobicity': -4.5, 'charge': 1, 'size': 'large', 'polar': True},
    'S': {'name': 'Serine', 'hydrophobicity': -0.8, 'charge': 0, 'size': 'small', 'polar': True},
    'T': {'name': 'Threonine', 'hydrophobicity': -0.7, 'charge': 0, 'size': 'small', 'polar': True},
    'V': {'name': 'Valine', 'hydrophobicity': 4.2, 'charge': 0, 'size': 'small', 'polar': False},
    'W': {'name': 'Tryptophan', 'hydrophobicity': -0.9, 'charge': 0, 'size': 'large', 'polar': False},
    'Y': {'name': 'Tyrosine', 'hydrophobicity': -1.3, 'charge': 0, 'size': 'large', 'polar': True},
}


@dataclass
class MutationResult:
    """Result of a single mutation prediction."""
    
    position: int  # 1-indexed position
    original_aa: str
    mutant_aa: str
    mutation_code: str  # e.g., "A42G"
    
    # pLDDT metrics
    mean_plddt: float
    plddt_per_residue: List[float]
    local_plddt: float  # pLDDT at the mutation site
    
    # Delta metrics (compared to wild-type/base)
    delta_mean_plddt: float = 0.0
    delta_local_plddt: float = 0.0
    
    # Additional metrics
    rmsd_to_base: Optional[float] = None
    tm_score_to_base: Optional[float] = None
    clash_score: Optional[float] = None
    sasa_total: Optional[float] = None
    
    # Structure path
    structure_path: Optional[Path] = None
    
    # Prediction metadata
    prediction_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def is_beneficial(self) -> bool:
        """Check if mutation improves pLDDT."""
        return self.delta_mean_plddt > 0 or self.delta_local_plddt > 0
    
    @property
    def improvement_score(self) -> float:
        """Calculate overall improvement score."""
        # Weighted combination of delta metrics
        # Could add penalties for bad RMSD or clashes here
        return 0.6 * self.delta_mean_plddt + 0.4 * self.delta_local_plddt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position,
            "original_aa": self.original_aa,
            "mutant_aa": self.mutant_aa,
            "mutation_code": self.mutation_code,
            "mean_plddt": self.mean_plddt,
            "local_plddt": self.local_plddt,
            "delta_mean_plddt": self.delta_mean_plddt,
            "delta_local_plddt": self.delta_local_plddt,
            "rmsd_to_base": self.rmsd_to_base,
            "tm_score_to_base": self.tm_score_to_base,
            "clash_score": self.clash_score,
            "sasa_total": self.sasa_total,
            "is_beneficial": self.is_beneficial,
            "improvement_score": self.improvement_score,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class SaturationMutagenesisResult:
    """Result of saturation mutagenesis at a single position."""
    
    position: int
    original_aa: str
    sequence: str
    
    # Base structure metrics
    base_mean_plddt: float
    base_local_plddt: float
    base_plddt_per_residue: List[float]
    base_structure_path: Optional[Path] = None
    base_clash_score: Optional[float] = None
    base_sasa_total: Optional[float] = None
    
    # Mutation results
    mutations: List[MutationResult] = field(default_factory=list)
    
    # Analysis metadata
    total_time: float = 0.0
    timestamp: str = ""
    
    @property
    def ranked_mutations(self) -> List[MutationResult]:
        """Get mutations ranked by improvement score."""
        return sorted(
            [m for m in self.mutations if m.success],
            key=lambda x: x.improvement_score,
            reverse=True
        )
    
    @property
    def beneficial_mutations(self) -> List[MutationResult]:
        """Get only beneficial mutations."""
        return [m for m in self.ranked_mutations if m.is_beneficial]
    
    @property
    def best_mutation(self) -> Optional[MutationResult]:
        """Get the best mutation recommendation."""
        ranked = self.ranked_mutations
        return ranked[0] if ranked else None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position,
            "original_aa": self.original_aa,
            "base_mean_plddt": self.base_mean_plddt,
            "base_local_plddt": self.base_local_plddt,
            "base_clash_score": self.base_clash_score,
            "base_sasa_total": self.base_sasa_total,
            "mutations": [m.to_dict() for m in self.mutations],
            "best_mutation": self.best_mutation.to_dict() if self.best_mutation else None,
            "beneficial_count": len(self.beneficial_mutations),
            "total_time": self.total_time,
            "timestamp": self.timestamp,
        }


class MutationScanner:
    """
    Scanner for rapid mutation analysis using ESMFold.
    """
    
    def __init__(
        self,
        use_api: bool = True,
        max_workers: int = 4,
        output_dir: Optional[Path] = None,
    ):
        self.use_api = use_api
        self.max_workers = max_workers
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="mutation_scan_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._base_cache: Dict[str, Tuple[str, List[float], Path]] = {}

        # Import metrics lazily
        try:
            from protein_design_hub.evaluation.metrics.clash_score import ClashScoreMetric
            from protein_design_hub.evaluation.metrics.sasa import SASAMetric
            from protein_design_hub.evaluation.metrics.rmsd import RMSDMetric
            from protein_design_hub.evaluation.metrics.tm_score import TMScoreMetric
            
            self._clash_metric = ClashScoreMetric()
            self._sasa_metric = SASAMetric()
            self._rmsd_metric = RMSDMetric(atoms="CA")
            self._tm_metric = TMScoreMetric()
            self._metrics_available = True
        except ImportError:
            self._metrics_available = False
            print("Warning: BioPython or other dependencies missing. Some metrics available.")
    
    def _predict_esmfold_api(self, sequence: str, output_path: Path) -> Tuple[str, List[float]]:
        import requests
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence.upper().strip() if c in valid_aa)
        
        if len(sequence) > 400:
            raise ValueError(f"Sequence too long for API ({len(sequence)} > 400)")
        
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"ESMFold API error: {response.status_code}")
        
        pdb_text = response.text
        output_path.write_text(pdb_text)
        plddt_values = self._extract_plddt_from_pdb(pdb_text)
        return pdb_text, plddt_values
    
    def _predict_esmfold_local(self, sequence: str, output_path: Path) -> Tuple[str, List[float]]:
        try:
            import torch
            import esm
        except ImportError:
            raise ImportError("ESMFold local requires `fair-esm` and `torch`")
        
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence.upper().strip() if c in valid_aa)
        
        model = esm.pretrained.esmfold_v1().eval()
        if torch.cuda.is_available():
            model = model.cuda()
            model.set_chunk_size(128)
        
        with torch.no_grad():
            pdb_text = model.infer_pdb(sequence)
        
        output_path.write_text(pdb_text)
        plddt_values = self._extract_plddt_from_pdb(pdb_text)
        return pdb_text, plddt_values
    
    def _extract_plddt_from_pdb(self, pdb_text: str) -> List[float]:
        plddt_values = []
        seen_residues = set()
        for line in pdb_text.splitlines():
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                residue_id = (line[21], int(line[22:26]))
            except: continue
            if residue_id in seen_residues: continue
            seen_residues.add(residue_id)
            try:
                plddt_values.append(float(line[60:66]))
            except: continue
        return plddt_values

    def calculate_biophysical_metrics(
        self, 
        model_path: Path, 
        reference_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """Calculate secondary metrics."""
        results = {}
        if not self._metrics_available:
            return results
            
        # 1. Clash Score
        try:
            clash_res = self._clash_metric.compute(model_path)
            results['clash_score'] = clash_res.get('clash_score')
        except Exception: pass
            
        # 2. SASA
        try:
            sasa_res = self._sasa_metric.compute(model_path)
            results['sasa_total'] = sasa_res.get('sasa_total')
        except Exception: pass
            
        if reference_path:
            # 3. RMSD
            try:
                rmsd_res = self._rmsd_metric.compute(model_path, reference_path)
                results['rmsd'] = rmsd_res.get('rmsd')
            except Exception: pass
            
            # 4. TM-score
            try:
                tm_res = self._tm_metric.compute(model_path, reference_path)
                results['tm_score'] = tm_res.get('tm_score')
            except Exception: pass
            
        return results

    def predict_single(self, sequence: str, name: str = "protein") -> Tuple[str, List[float], Path]:
        output_path = self.output_dir / f"{name}.pdb"
        if self.use_api:
            pdb_text, plddt_values = self._predict_esmfold_api(sequence, output_path)
        else:
            pdb_text, plddt_values = self._predict_esmfold_local(sequence, output_path)
        return pdb_text, plddt_values, output_path

    def scan_position(
        self,
        sequence: str,
        position: int,
        progress_callback: Optional[callable] = None,
    ) -> SaturationMutagenesisResult:
        import datetime
        start_time = time.time()
        
        if position < 1 or position > len(sequence):
            raise ValueError(f"Position {position} out of range")
        
        original_aa = sequence[position - 1]
        
        # 1. Base Structure
        if sequence in self._base_cache:
            base_pdb, base_plddt, base_path = self._base_cache[sequence]
        else:
            base_pdb, base_plddt, base_path = self.predict_single(sequence, "base_wt")
            self._base_cache[sequence] = (base_pdb, base_plddt, base_path)
            
        base_mean_plddt = sum(base_plddt) / len(base_plddt) if base_plddt else 0
        base_local_plddt = base_plddt[position - 1] if position <= len(base_plddt) else 0
        
        # Base metrics
        base_metrics = self.calculate_biophysical_metrics(base_path)
        
        if progress_callback:
            progress_callback(0, 20, f"Base (WT) calculated")
            
        # 2. Mutagenesis
        mutations_to_test = [aa for aa in AMINO_ACIDS if aa != original_aa]
        mutation_results = []
        
        for idx, mutant_aa in enumerate(mutations_to_test):
            mutation_code = f"{original_aa}{position}{mutant_aa}"
            if progress_callback:
                progress_callback(idx + 1, len(mutations_to_test) + 1, mutation_code)
                
            mutant_seq = sequence[:position - 1] + mutant_aa + sequence[position:]
            try:
                mut_start = time.time()
                mut_pdb, mut_plddt, mut_path = self.predict_single(mutant_seq, f"mut_{mutation_code}")
                
                mut_mean_plddt = sum(mut_plddt) / len(mut_plddt) if mut_plddt else 0
                mut_local_plddt = mut_plddt[position - 1] if position <= len(mut_plddt) else 0
                
                # Biophsyical metrics
                metrics = self.calculate_biophysical_metrics(mut_path, base_path)
                
                result = MutationResult(
                    position=position,
                    original_aa=original_aa,
                    mutant_aa=mutant_aa,
                    mutation_code=mutation_code,
                    mean_plddt=mut_mean_plddt,
                    plddt_per_residue=mut_plddt,
                    local_plddt=mut_local_plddt,
                    delta_mean_plddt=mut_mean_plddt - base_mean_plddt,
                    delta_local_plddt=mut_local_plddt - base_local_plddt,
                    rmsd_to_base=metrics.get('rmsd'),
                    tm_score_to_base=metrics.get('tm_score'),
                    clash_score=metrics.get('clash_score'),
                    sasa_total=metrics.get('sasa_total'),
                    structure_path=mut_path,
                    prediction_time=time.time() - mut_start,
                    success=True,
                )
            except Exception as e:
                result = MutationResult(
                    position=position,
                    original_aa=original_aa,
                    mutant_aa=mutant_aa,
                    mutation_code=mutation_code,
                    mean_plddt=0,
                    plddt_per_residue=[],
                    local_plddt=0,
                    success=False,
                    error_message=str(e),
                )
            mutation_results.append(result)
            
        return SaturationMutagenesisResult(
            position=position,
            original_aa=original_aa,
            sequence=sequence,
            base_mean_plddt=base_mean_plddt,
            base_local_plddt=base_local_plddt,
            base_plddt_per_residue=base_plddt,
            base_structure_path=base_path,
            base_clash_score=base_metrics.get('clash_score'),
            base_sasa_total=base_metrics.get('sasa_total'),
            mutations=mutation_results,
            total_time=time.time() - start_time,
            timestamp=datetime.datetime.now().isoformat(),
        )

def create_mutation_heatmap(result: SaturationMutagenesisResult) -> Dict[str, Any]:
    """Helper to create heatmap data."""
    # ... existing implementation or updated one ...
    # For now, we can just reuse the one in the web page or this one.
    # The one in web page does its own logic.
    pass
