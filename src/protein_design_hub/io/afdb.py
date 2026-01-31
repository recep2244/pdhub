"""AlphaFold DB (AFDB) lookup utilities via EBI BLAST and AFDB API."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import requests


EBI_BLAST_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"
AFDB_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction"


@dataclass
class BlastHit:
    uniprot_id: str
    identity: float
    coverage: float
    evalue: Optional[float]
    query_length: int
    align_length: int


@dataclass
class AFDBMatch:
    uniprot_id: str
    identity: float
    coverage: float
    evalue: Optional[float]
    entry_id: Optional[str]
    pdb_url: Optional[str]
    cif_url: Optional[str]
    pae_doc_url: Optional[str]
    pae_image_url: Optional[str]
    prediction: Dict[str, Any]
    structure_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.structure_path:
            payload["structure_path"] = str(self.structure_path)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AFDBMatch":
        structure_path = payload.get("structure_path")
        if structure_path:
            payload = dict(payload)
            payload["structure_path"] = Path(structure_path)
        return cls(**payload)


def normalize_sequence(sequence: str) -> str:
    """Normalize a FASTA-like sequence to a single chain suitable for BLAST."""
    if not sequence:
        return ""
    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    seq_parts = [line for line in lines if not line.startswith(">")]
    raw = "".join(seq_parts)
    if ":" in raw:
        chains = [c for c in raw.split(":") if c]
        if chains:
            raw = max(chains, key=len)
    valid = set("ACDEFGHIKLMNPQRSTVWYX")
    return "".join([c for c in raw.upper() if c in valid])


def _interval_union(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            total += max(0, cur_end - cur_start + 1)
            cur_start, cur_end = start, end
    total += max(0, cur_end - cur_start + 1)
    return total


def _parse_uniprot_id(hit_id: str, hit_accession: Optional[str]) -> Optional[str]:
    if hit_accession:
        return hit_accession
    if not hit_id:
        return None
    # Typical BLAST hit IDs look like: sp|P12345|NAME
    if "|" in hit_id:
        parts = hit_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return hit_id


def _parse_blast_xml(xml_text: str) -> List[BlastHit]:
    root = ElementTree.fromstring(xml_text)
    query_len = int(root.findtext(".//BlastOutput_query-len", default="0"))
    hits: List[BlastHit] = []

    for hit in root.findall(".//Hit"):
        hit_id = hit.findtext("Hit_id") or ""
        hit_accession = hit.findtext("Hit_accession")
        uniprot_id = _parse_uniprot_id(hit_id, hit_accession)
        if not uniprot_id:
            continue

        best_identity = 0.0
        best_evalue: Optional[float] = None
        align_len = 0
        intervals: List[Tuple[int, int]] = []

        for hsp in hit.findall(".//Hsp"):
            try:
                ident = int(hsp.findtext("Hsp_identity", default="0"))
                aln = int(hsp.findtext("Hsp_align-len", default="0"))
            except ValueError:
                continue
            if aln > 0:
                pct = (ident / aln) * 100.0
                if pct > best_identity:
                    best_identity = pct
                    align_len = aln
            try:
                qstart = int(hsp.findtext("Hsp_query-from", default="0"))
                qend = int(hsp.findtext("Hsp_query-to", default="0"))
            except ValueError:
                qstart = 0
                qend = 0
            if qstart and qend:
                intervals.append((min(qstart, qend), max(qstart, qend)))
            try:
                evalue = float(hsp.findtext("Hsp_evalue", default="inf"))
                if best_evalue is None or evalue < best_evalue:
                    best_evalue = evalue
            except ValueError:
                pass

        coverage = 0.0
        if query_len > 0 and intervals:
            covered = _interval_union(intervals)
            coverage = (covered / query_len) * 100.0

        hits.append(
            BlastHit(
                uniprot_id=uniprot_id,
                identity=best_identity,
                coverage=coverage,
                evalue=best_evalue,
                query_length=query_len,
                align_length=align_len,
            )
        )

    return hits


def _blast_result_types(job_id: str, timeout: int = 30) -> str:
    url = f"{EBI_BLAST_URL}/resulttypes/{job_id}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _blast_result(job_id: str, result_type: str, timeout: int = 60) -> str:
    url = f"{EBI_BLAST_URL}/result/{job_id}/{result_type}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def run_blast(
    sequence: str,
    email: str,
    program: str = "blastp",
    database: str = "uniprotkb",
    stype: str = "protein",
    poll_interval: int = 5,
    max_wait: int = 300,
) -> List[BlastHit]:
    payload = {
        "email": email,
        "program": program,
        "database": database,
        "stype": stype,
        "sequence": sequence,
    }
    response = requests.post(f"{EBI_BLAST_URL}/run", data=payload, timeout=30)
    response.raise_for_status()
    job_id = response.text.strip()
    if not job_id:
        raise RuntimeError("BLAST run did not return a job id")

    status_url = f"{EBI_BLAST_URL}/status/{job_id}"
    start = time.time()
    status = "RUNNING"
    while time.time() - start < max_wait:
        status_resp = requests.get(status_url, timeout=15)
        status_resp.raise_for_status()
        status = status_resp.text.strip().upper()
        if status == "FINISHED":
            break
        if status in {"ERROR", "FAILURE", "NOT_FOUND"}:
            raise RuntimeError(f"BLAST failed with status {status}")
        time.sleep(poll_interval)

    if status != "FINISHED":
        raise TimeoutError("BLAST search timed out")

    result_types = _blast_result_types(job_id)
    result_types_lower = result_types.lower()
    xml_type = "xml" if "xml" in result_types_lower else None

    if xml_type:
        xml_text = _blast_result(job_id, "xml")
        return _parse_blast_xml(xml_text)

    out_text = _blast_result(job_id, "out")
    raise RuntimeError(
        "BLAST results returned without XML output; unable to parse hits."
    )


class AFDBClient:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (Path.home() / ".pdhub" / "cache" / "afdb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_structure(self, url: str, filename: str) -> Path:
        path = self.cache_dir / filename
        if path.exists():
            return path
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        path.write_text(resp.text)
        return path

    def fetch_prediction(self, uniprot_id: str) -> Optional[Dict[str, Any]]:
        url = f"{AFDB_PREDICTION_URL}/{uniprot_id}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        return data[0]

    def find_match(
        self,
        sequence: str,
        min_identity: float = 90.0,
        min_coverage: float = 90.0,
        email: Optional[str] = None,
        max_wait: int = 300,
    ) -> Tuple[Optional[AFDBMatch], Optional[str]]:
        clean_seq = normalize_sequence(sequence)
        if not clean_seq:
            return None, "No valid sequence provided"

        email = email or os.getenv("EBI_EMAIL") or os.getenv("AFDB_BLAST_EMAIL") or "pdhub@local"

        try:
            hits = run_blast(clean_seq, email=email, max_wait=max_wait)
        except Exception as exc:
            return None, str(exc)

        if not hits:
            return None, None

        qualified = [
            hit for hit in hits
            if hit.identity >= min_identity and hit.coverage >= min_coverage
        ]
        if not qualified:
            return None, None

        qualified.sort(
            key=lambda h: (
                -h.identity,
                -h.coverage,
                h.evalue if h.evalue is not None else float("inf"),
            )
        )
        top = qualified[0]

        prediction = self.fetch_prediction(top.uniprot_id)
        if not prediction:
            return None, None

        pdb_url = prediction.get("pdbUrl") or prediction.get("pdb_url")
        cif_url = prediction.get("cifUrl") or prediction.get("cif_url") or prediction.get("bcifUrl")
        pae_doc_url = prediction.get("paeDocUrl") or prediction.get("pae_doc_url")
        pae_image_url = prediction.get("paeImageUrl") or prediction.get("pae_image_url")
        entry_id = prediction.get("entryId") or prediction.get("entry_id")

        structure_path = None
        if pdb_url:
            filename = f"{entry_id or top.uniprot_id}.pdb"
            structure_path = self._download_structure(pdb_url, filename)
        elif cif_url:
            filename = f"{entry_id or top.uniprot_id}.cif"
            structure_path = self._download_structure(cif_url, filename)

        match = AFDBMatch(
            uniprot_id=top.uniprot_id,
            identity=top.identity,
            coverage=top.coverage,
            evalue=top.evalue,
            entry_id=entry_id,
            pdb_url=pdb_url,
            cif_url=cif_url,
            pae_doc_url=pae_doc_url,
            pae_image_url=pae_image_url,
            prediction=prediction,
            structure_path=structure_path,
        )
        return match, None

    def cache_key(self, sequence: str) -> str:
        clean_seq = normalize_sequence(sequence)
        digest = hashlib.sha1(clean_seq.encode("utf-8")).hexdigest()
        return f"afdb:{digest}"
