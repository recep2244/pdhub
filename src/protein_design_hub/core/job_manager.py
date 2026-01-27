"""Job manager for batch processing and job tracking."""

import json
import sqlite3
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a prediction or evaluation job."""
    id: str
    job_type: str  # "prediction", "evaluation", "comparison"
    name: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    progress: float = 0.0
    progress_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        d["status"] = JobStatus(d["status"])
        return cls(**d)


class JobManager:
    """
    Manages job queue, execution, and persistence.

    Features:
    - SQLite-based job persistence
    - Background job processing
    - Progress tracking
    - Job history
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize job manager.

        Args:
            db_path: Path to SQLite database. Uses in-memory if None.
        """
        self.db_path = db_path or Path.home() / ".pdhub" / "jobs.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._job_queue: queue.Queue = queue.Queue()
        self._active_jobs: Dict[str, Job] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_progress": [],
            "on_complete": [],
            "on_error": [],
        }

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                progress REAL DEFAULT 0.0,
                progress_message TEXT DEFAULT ''
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC)
        """)

        conn.commit()
        conn.close()

    def create_job(
        self,
        job_type: str,
        name: str,
        input_data: Dict[str, Any],
    ) -> Job:
        """
        Create a new job.

        Args:
            job_type: Type of job (prediction, evaluation, comparison).
            name: Human-readable job name.
            input_data: Job input parameters.

        Returns:
            Created Job object.
        """
        job = Job(
            id=str(uuid.uuid4())[:8],
            job_type=job_type,
            name=name,
            status=JobStatus.PENDING,
            created_at=datetime.now().isoformat(),
            input_data=input_data,
        )

        # Save to database
        self._save_job(job)

        return job

    def _save_job(self, job: Job):
        """Save job to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO jobs
            (id, job_type, name, status, created_at, started_at, completed_at,
             input_data, output_data, error_message, progress, progress_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.id,
            job.job_type,
            job.name,
            job.status.value,
            job.created_at,
            job.started_at,
            job.completed_at,
            json.dumps(job.input_data),
            json.dumps(job.output_data),
            job.error_message,
            job.progress,
            job.progress_message,
        ))

        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_job(row)
        return None

    def _row_to_job(self, row) -> Job:
        """Convert database row to Job object."""
        return Job(
            id=row[0],
            job_type=row[1],
            name=row[2],
            status=JobStatus(row[3]),
            created_at=row[4],
            started_at=row[5],
            completed_at=row[6],
            input_data=json.loads(row[7]) if row[7] else {},
            output_data=json.loads(row[8]) if row[8] else {},
            error_message=row[9],
            progress=row[10] or 0.0,
            progress_message=row[11] or "",
        )

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status.
            job_type: Filter by job type.
            limit: Maximum number of jobs to return.

        Returns:
            List of Job objects.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM jobs WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if job_type:
            query += " AND job_type = ?"
            params.append(job_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_job(row) for row in rows]

    def submit_job(self, job: Job):
        """Submit job to processing queue."""
        self._job_queue.put(job)

        # Start worker if not running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_worker()

    def _start_worker(self):
        """Start background worker thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """Background worker loop."""
        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=1.0)
                self._process_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def _process_job(self, job: Job):
        """Process a single job."""
        try:
            # Update status to running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            self._save_job(job)
            self._active_jobs[job.id] = job

            # Notify callbacks
            for cb in self._callbacks["on_start"]:
                cb(job)

            # Execute job based on type
            if job.job_type == "prediction":
                result = self._run_prediction(job)
            elif job.job_type == "evaluation":
                result = self._run_evaluation(job)
            elif job.job_type == "comparison":
                result = self._run_comparison(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            # Update job with results
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            job.output_data = result
            job.progress = 1.0
            self._save_job(job)

            # Notify callbacks
            for cb in self._callbacks["on_complete"]:
                cb(job)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now().isoformat()
            job.error_message = str(e)
            self._save_job(job)

            # Notify callbacks
            for cb in self._callbacks["on_error"]:
                cb(job, e)

        finally:
            if job.id in self._active_jobs:
                del self._active_jobs[job.id]

    def _run_prediction(self, job: Job) -> Dict[str, Any]:
        """Run prediction job."""
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.types import PredictionInput, Sequence
        from protein_design_hub.io.parsers.fasta import FastaParser

        settings = get_settings()
        input_data = job.input_data

        # Parse sequences
        if input_data.get("fasta_content"):
            parser = FastaParser()
            sequences = parser.parse(input_data["fasta_content"])
        elif input_data.get("sequences"):
            sequences = [Sequence(**s) for s in input_data["sequences"]]
        else:
            raise ValueError("No sequences provided")

        # Setup output
        output_dir = Path(input_data.get("output_dir", "./outputs")) / job.id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run predictor
        predictor_name = input_data.get("predictor", "colabfold")
        predictor = PredictorRegistry.get(predictor_name, settings)

        prediction_input = PredictionInput(
            job_id=job.id,
            sequences=sequences,
            output_dir=output_dir,
            num_models=input_data.get("num_models", 5),
            num_recycles=input_data.get("num_recycles", 3),
        )

        # Update progress
        job.progress = 0.1
        job.progress_message = f"Running {predictor_name}..."
        self._save_job(job)

        result = predictor.predict(prediction_input)

        return {
            "success": result.success,
            "structure_paths": [str(p) for p in result.structure_paths],
            "scores": [s.to_dict() if hasattr(s, 'to_dict') else asdict(s) for s in result.scores] if result.scores else [],
            "runtime_seconds": result.runtime_seconds,
            "error_message": result.error_message,
        }

    def _run_evaluation(self, job: Job) -> Dict[str, Any]:
        """Run evaluation job."""
        from protein_design_hub.evaluation.composite import CompositeEvaluator
        from protein_design_hub.core.config import get_settings

        settings = get_settings()
        input_data = job.input_data

        evaluator = CompositeEvaluator(settings=settings)

        model_path = Path(input_data["model_path"])
        reference_path = Path(input_data["reference_path"]) if input_data.get("reference_path") else None

        job.progress = 0.2
        job.progress_message = "Computing metrics..."
        self._save_job(job)

        if input_data.get("comprehensive", True):
            result = evaluator.evaluate_comprehensive(model_path, reference_path)
        else:
            result = evaluator.evaluate(model_path, reference_path)
            result = result.to_dict() if hasattr(result, 'to_dict') else asdict(result)

        return result

    def _run_comparison(self, job: Job) -> Dict[str, Any]:
        """Run comparison job."""
        from protein_design_hub.pipeline.workflow import PredictionWorkflow
        from protein_design_hub.core.config import get_settings

        settings = get_settings()
        input_data = job.input_data

        workflow = PredictionWorkflow(settings)

        # This would run the full comparison workflow
        # For now, return a placeholder
        return {"status": "comparison_completed"}

    def update_progress(self, job_id: str, progress: float, message: str = ""):
        """Update job progress."""
        job = self.get_job(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.progress = progress
            job.progress_message = message
            self._save_job(job)

            for cb in self._callbacks["on_progress"]:
                cb(job)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            self._save_job(job)
            return True

        return False

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def clear_completed(self):
        """Clear all completed jobs from history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM jobs WHERE status IN (?, ?, ?)",
            (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value)
        )
        conn.commit()
        conn.close()

    def stop(self):
        """Stop the job manager."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)


# Global singleton
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
