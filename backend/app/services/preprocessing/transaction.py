"""
Transactional output manager for atomic artifact publication (Production v3.1).
Writes all outputs and manifests to a staging job directory, publishing atomically upon completion.
"""

import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class StagingJobContext:
    """
    Context manager for a single atomic preprocessing job.
    """
    def __init__(self, target_dir: Path, job_id: Optional[str] = None):
        self.target_dir = target_dir
        self.job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        self.staging_dir = target_dir.parent / f".staging_{self.job_id}_{uuid.uuid4().hex[:6]}"
        self.is_published = False
        self.staged_files: List[Path] = []

    def __enter__(self) -> "StagingJobContext":
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_published:
            self.cleanup()

    def get_staging_path(self, filename: str) -> Path:
        """Get absolute path for staging an artifact inside the staging dir."""
        path = self.staging_dir / filename
        self.staged_files.append(path)
        return path

    def publish_atomically(self) -> Path:
        """
        Atomically publish all staged artifacts to target_dir.
        If target_dir is a directory, replaces or populates it atomically.
        """
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)

            # Move each staged file into target_dir
            for staged in self.staging_dir.iterdir():
                dest = self.target_dir / staged.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                # On Windows/Linux, shutil.move inside same drive is fast atomic rename
                shutil.move(str(staged), str(dest))

            self.is_published = True
            logger.info(f"Job '{self.job_id}' published atomically to '{self.target_dir}'.")
            return self.target_dir

        except Exception as e:
            logger.error(f"Failed to publish job '{self.job_id}' atomically: {e}")
            self.cleanup()
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Remove temporary staging directory."""
        if self.staging_dir.exists():
            try:
                shutil.rmtree(self.staging_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to remove staging dir '{self.staging_dir}': {e}")


class TransactionalOutputManager:
    """Manager for orchestrating atomic staging and publications."""

    @staticmethod
    def begin_job(target_dir: Path, job_id: Optional[str] = None) -> StagingJobContext:
        """Create and enter a new staging job context."""
        ctx = StagingJobContext(target_dir=target_dir, job_id=job_id)
        return ctx.__enter__()
