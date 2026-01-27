"""Utilities for fetching protein data from external databases."""

import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import tempfile
import re


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    success: bool
    data: Optional[str] = None
    file_path: Optional[Path] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PDBFetcher:
    """Fetch protein structures from RCSB PDB."""

    BASE_URL = "https://files.rcsb.org/download"
    API_URL = "https://data.rcsb.org/rest/v1/core/entry"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize PDB fetcher.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "pdhub_cache" / "pdb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_structure(
        self,
        pdb_id: str,
        file_format: str = "pdb",
        save_path: Optional[Path] = None,
    ) -> FetchResult:
        """
        Fetch a protein structure from PDB.

        Args:
            pdb_id: 4-letter PDB ID (e.g., "1abc").
            file_format: Output format ("pdb", "cif", "mmcif").
            save_path: Optional path to save the file.

        Returns:
            FetchResult with structure data.
        """
        pdb_id = pdb_id.lower().strip()

        if not re.match(r'^[a-z0-9]{4}$', pdb_id):
            return FetchResult(
                success=False,
                error=f"Invalid PDB ID format: {pdb_id}. Expected 4-character alphanumeric code."
            )

        # Map format to extension
        format_map = {
            "pdb": "pdb",
            "cif": "cif",
            "mmcif": "cif",
        }
        ext = format_map.get(file_format.lower(), "pdb")

        # Check cache first
        cache_path = self.cache_dir / f"{pdb_id}.{ext}"
        if cache_path.exists():
            return FetchResult(
                success=True,
                data=cache_path.read_text(),
                file_path=cache_path,
                metadata={"source": "cache", "pdb_id": pdb_id},
            )

        # Fetch from PDB
        url = f"{self.BASE_URL}/{pdb_id}.{ext}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            content = response.text

            # Save to cache
            cache_path.write_text(content)

            # Save to custom path if provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(content)
                file_path = save_path
            else:
                file_path = cache_path

            return FetchResult(
                success=True,
                data=content,
                file_path=file_path,
                metadata={"source": "rcsb", "pdb_id": pdb_id, "url": url},
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return FetchResult(
                    success=False,
                    error=f"PDB ID '{pdb_id}' not found."
                )
            return FetchResult(
                success=False,
                error=f"HTTP error fetching {pdb_id}: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Network error fetching {pdb_id}: {str(e)}"
            )

    def get_metadata(self, pdb_id: str) -> FetchResult:
        """
        Get metadata for a PDB entry.

        Args:
            pdb_id: 4-letter PDB ID.

        Returns:
            FetchResult with metadata dictionary.
        """
        pdb_id = pdb_id.lower().strip()

        try:
            url = f"{self.API_URL}/{pdb_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract useful info
            metadata = {
                "pdb_id": pdb_id,
                "title": data.get("struct", {}).get("title", ""),
                "method": data.get("exptl", [{}])[0].get("method", ""),
                "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                "deposition_date": data.get("rcsb_accession_info", {}).get("deposit_date", ""),
                "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date", ""),
                "organism": data.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0),
            }

            return FetchResult(
                success=True,
                metadata=metadata,
            )

        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Error fetching metadata for {pdb_id}: {str(e)}"
            )

    def search(self, query: str, limit: int = 10) -> FetchResult:
        """
        Search PDB for structures.

        Args:
            query: Search query (protein name, organism, etc.).
            limit: Maximum number of results.

        Returns:
            FetchResult with list of matching PDB IDs.
        """
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        search_query = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {
                    "value": query
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": limit
                }
            }
        }

        try:
            response = requests.post(search_url, json=search_query, timeout=15)
            response.raise_for_status()

            data = response.json()
            results = [hit["identifier"] for hit in data.get("result_set", [])]

            return FetchResult(
                success=True,
                metadata={"query": query, "results": results, "total": data.get("total_count", 0)},
            )

        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Search error: {str(e)}"
            )


class UniProtFetcher:
    """Fetch protein sequences and data from UniProt."""

    BASE_URL = "https://rest.uniprot.org"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize UniProt fetcher.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "pdhub_cache" / "uniprot"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_sequence(
        self,
        accession: str,
        include_isoforms: bool = False,
    ) -> FetchResult:
        """
        Fetch protein sequence from UniProt.

        Args:
            accession: UniProt accession (e.g., "P12345" or "EGFR_HUMAN").
            include_isoforms: Include isoform sequences.

        Returns:
            FetchResult with FASTA sequence.
        """
        accession = accession.strip()

        # Check cache
        cache_path = self.cache_dir / f"{accession}.fasta"
        if cache_path.exists():
            return FetchResult(
                success=True,
                data=cache_path.read_text(),
                file_path=cache_path,
                metadata={"source": "cache", "accession": accession},
            )

        # Fetch from UniProt
        url = f"{self.BASE_URL}/uniprotkb/{accession}.fasta"
        if include_isoforms:
            url += "?includeIsoform=true"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            content = response.text

            # Cache
            cache_path.write_text(content)

            return FetchResult(
                success=True,
                data=content,
                file_path=cache_path,
                metadata={"source": "uniprot", "accession": accession},
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return FetchResult(
                    success=False,
                    error=f"UniProt accession '{accession}' not found."
                )
            return FetchResult(
                success=False,
                error=f"HTTP error: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Network error: {str(e)}"
            )

    def get_entry(self, accession: str) -> FetchResult:
        """
        Get full UniProt entry data.

        Args:
            accession: UniProt accession.

        Returns:
            FetchResult with entry data.
        """
        url = f"{self.BASE_URL}/uniprotkb/{accession}.json"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            # Extract key information
            metadata = {
                "accession": data.get("primaryAccession", accession),
                "name": data.get("uniProtkbId", ""),
                "protein_name": self._get_protein_name(data),
                "organism": data.get("organism", {}).get("scientificName", ""),
                "length": data.get("sequence", {}).get("length", 0),
                "sequence": data.get("sequence", {}).get("value", ""),
                "function": self._get_function(data),
                "subcellular_location": self._get_subcellular_location(data),
                "pdb_ids": self._get_pdb_ids(data),
            }

            return FetchResult(
                success=True,
                metadata=metadata,
            )

        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Error fetching entry: {str(e)}"
            )

    def _get_protein_name(self, data: Dict) -> str:
        """Extract protein name from UniProt data."""
        try:
            return data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except (KeyError, TypeError):
            try:
                return data["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
            except (KeyError, TypeError, IndexError):
                return ""

    def _get_function(self, data: Dict) -> str:
        """Extract function description."""
        try:
            comments = data.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        return texts[0].get("value", "")
        except (KeyError, TypeError):
            pass
        return ""

    def _get_subcellular_location(self, data: Dict) -> List[str]:
        """Extract subcellular locations."""
        locations = []
        try:
            comments = data.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "SUBCELLULAR LOCATION":
                    for loc in comment.get("subcellularLocations", []):
                        location = loc.get("location", {}).get("value", "")
                        if location:
                            locations.append(location)
        except (KeyError, TypeError):
            pass
        return locations

    def _get_pdb_ids(self, data: Dict) -> List[str]:
        """Extract associated PDB IDs."""
        pdb_ids = []
        try:
            for ref in data.get("uniProtKBCrossReferences", []):
                if ref.get("database") == "PDB":
                    pdb_ids.append(ref.get("id", ""))
        except (KeyError, TypeError):
            pass
        return pdb_ids

    def search(
        self,
        query: str,
        organism: Optional[str] = None,
        limit: int = 10,
    ) -> FetchResult:
        """
        Search UniProt for proteins.

        Args:
            query: Search query.
            organism: Optional organism filter.
            limit: Maximum results.

        Returns:
            FetchResult with search results.
        """
        search_query = query
        if organism:
            search_query += f" AND organism_name:{organism}"

        url = f"{self.BASE_URL}/uniprotkb/search"
        params = {
            "query": search_query,
            "format": "json",
            "size": limit,
            "fields": "accession,id,protein_name,organism_name,length",
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            results = []
            for entry in data.get("results", []):
                results.append({
                    "accession": entry.get("primaryAccession", ""),
                    "name": entry.get("uniProtkbId", ""),
                    "protein_name": self._get_protein_name(entry),
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                    "length": entry.get("sequence", {}).get("length", 0),
                })

            return FetchResult(
                success=True,
                metadata={"query": query, "results": results},
            )

        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Search error: {str(e)}"
            )


class AlphaFoldDBFetcher:
    """Fetch predicted structures from AlphaFold Database."""

    BASE_URL = "https://alphafold.ebi.ac.uk/files"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize AlphaFold DB fetcher.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "pdhub_cache" / "alphafold"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_structure(
        self,
        uniprot_id: str,
        version: int = 4,
        save_path: Optional[Path] = None,
    ) -> FetchResult:
        """
        Fetch AlphaFold predicted structure.

        Args:
            uniprot_id: UniProt accession.
            version: AlphaFold DB version (default: 4).
            save_path: Optional path to save structure.

        Returns:
            FetchResult with structure data.
        """
        uniprot_id = uniprot_id.upper().strip()

        # Check cache
        cache_path = self.cache_dir / f"AF-{uniprot_id}-F1-model_v{version}.pdb"
        if cache_path.exists():
            return FetchResult(
                success=True,
                data=cache_path.read_text(),
                file_path=cache_path,
                metadata={"source": "cache", "uniprot_id": uniprot_id, "version": version},
            )

        # Fetch from AlphaFold DB
        url = f"{self.BASE_URL}/AF-{uniprot_id}-F1-model_v{version}.pdb"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            content = response.text

            # Cache
            cache_path.write_text(content)

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(content)
                file_path = save_path
            else:
                file_path = cache_path

            return FetchResult(
                success=True,
                data=content,
                file_path=file_path,
                metadata={"source": "alphafold_db", "uniprot_id": uniprot_id, "version": version},
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return FetchResult(
                    success=False,
                    error=f"AlphaFold structure for '{uniprot_id}' not found in database."
                )
            return FetchResult(
                success=False,
                error=f"HTTP error: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Network error: {str(e)}"
            )

    def fetch_pae(
        self,
        uniprot_id: str,
        version: int = 4,
    ) -> FetchResult:
        """
        Fetch PAE (Predicted Aligned Error) data.

        Args:
            uniprot_id: UniProt accession.
            version: AlphaFold DB version.

        Returns:
            FetchResult with PAE JSON data.
        """
        uniprot_id = uniprot_id.upper().strip()

        url = f"{self.BASE_URL}/AF-{uniprot_id}-F1-predicted_aligned_error_v{version}.json"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            return FetchResult(
                success=True,
                metadata={"pae": data, "uniprot_id": uniprot_id},
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return FetchResult(
                    success=False,
                    error=f"PAE data for '{uniprot_id}' not found."
                )
            return FetchResult(
                success=False,
                error=f"HTTP error: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            return FetchResult(
                success=False,
                error=f"Network error: {str(e)}"
            )


def parse_fasta(fasta_content: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA format content.

    Args:
        fasta_content: FASTA format string.

    Returns:
        List of (header, sequence) tuples.
    """
    sequences = []
    current_header = None
    current_seq = []

    for line in fasta_content.strip().split('\n'):
        if line.startswith('>'):
            if current_header and current_seq:
                sequences.append((current_header, ''.join(current_seq)))
            current_header = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line.strip())

    if current_header and current_seq:
        sequences.append((current_header, ''.join(current_seq)))

    return sequences
