"""Run the planned training with IEMOCAP restored from its stored copy.

    python -m compact_ser.training_job --store-dir DRIVE/datasets/iemocap --search-dir DRIVE \
        --local-dir /content/compact-ser-data --output-dir DRIVE/study --scope core --max-runs 4

The stored study archive is copied to local disk and verified, and the four-class counts are
checked. Pending runs are then executed in plan order. Run folders and the run ledger are written
to the output folder, so the next session continues with the next pending run; an interrupted run
starts again from the beginning. `--runs` restricts this session to the named planned runs. A lock
file in the output folder stops two runtimes from training the same study at once.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from pathlib import Path

from compact_ser.corpora import read_iemocap
from compact_ser.iemocap_archive import check_corpus, locate_study_store, restore_study_files
from compact_ser.run_ledger import RunLedger
from compact_ser.runtime_environment import save_runtime_record
from compact_ser.study_design import PlanScope, RunSpec, build_run_plan
from compact_ser.study_runner import (
    LEDGER_FILE_NAME,
    add_check_options,
    build_context,
    check_study_settings,
    execute_pending_runs,
    study_design,
)

COMPLETION_MARKER = "TRAINING JOB COMPLETE"
FAILURE_MARKER = "TRAINING JOB FAILED"
LOCK_FILE_NAME = "study.lock"


class StudyLockedError(RuntimeError):
    """Raised when another runtime is training the same study."""


class StudyLock:
    """Marks a study folder as in use by one runtime.

    The lock file holds its owner and a time that a background thread refreshes. A lock that has
    not been refreshed for `stale_after` seconds belongs to a runtime that has stopped, and is
    taken over.

    Args:
        path: The lock file.
        stale_after: Seconds after which an unrefreshed lock is ignored.
        refresh_interval: Seconds between refreshes.
    """

    def __init__(self, path: Path, stale_after: float = 600.0, refresh_interval: float = 60.0):
        self.path = path
        self.stale_after = stale_after
        self.refresh_interval = refresh_interval
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> dict[str, object] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self) -> None:
        record = {"owner": self.owner, "refreshed_at": time.time()}
        self.path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def _refresh(self) -> None:
        while not self._stop.wait(self.refresh_interval):
            # A dropped Drive connection is retried at the next refresh.
            with contextlib.suppress(OSError):
                self._write()

    def __enter__(self) -> StudyLock:
        """Take the lock.

        Raises:
            StudyLockedError: If another runtime refreshed the lock recently.
        """
        record = self._read()
        if record is not None:
            age = time.time() - float(record["refreshed_at"])
            if age < self.stale_after:
                raise StudyLockedError(
                    f"Another runtime ({record['owner']}) is training this study; it last "
                    f"reported {age:.0f} s ago. If that runtime has stopped, wait "
                    f"{self.stale_after - age:.0f} s or delete {self.path}."
                )
            print(f"Taking over a lock last refreshed {age:.0f} s ago.", flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exception: object) -> None:
        """Release the lock if this runtime still owns it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        record = self._read()
        if record is not None and record["owner"] == self.owner:
            self.path.unlink(missing_ok=True)


def select_runs(plan: Sequence[RunSpec], run_ids: Sequence[str] | None) -> list[RunSpec]:
    """Keep the planned runs with the given identifiers, in plan order.

    Args:
        plan: Run specifications in execution order.
        run_ids: Identifiers to keep, or `None` to keep every run.

    Returns:
        The selected runs.

    Raises:
        ValueError: If an identifier is not in the plan.
    """
    if not run_ids:
        return list(plan)
    unknown = sorted(set(run_ids) - {run.run_id for run in plan})
    if unknown:
        raise ValueError(f"These runs are not in the plan: {', '.join(unknown)}")
    return [run for run in plan if run.run_id in set(run_ids)]


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse command-line arguments, restore the corpus, and execute pending runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument(
        "--search-dir", type=Path, help="Folder to search if the files are not in --store-dir."
    )
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=[scope.value for scope in PlanScope], default="core")
    parser.add_argument("--runs", nargs="+", metavar="RUN_ID", help="Run only these planned runs.")
    parser.add_argument("--max-runs", type=int, help="Largest number of runs in this session.")
    add_check_options(parser)
    options = parser.parse_args(arguments)

    try:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Environment recorded in {save_runtime_record(options.output_dir)}.", flush=True)
        design = study_design(options.update_limit)
        check_study_settings(options.output_dir, design)
        store_dir = locate_study_store(options.store_dir, options.search_dir)
        corpus_dir = restore_study_files(store_dir, options.local_dir)
        print(f"Four-class utterances: {check_corpus(corpus_dir)['emotions']}", flush=True)

        context = build_context(
            read_iemocap(corpus_dir, design.sessions),
            corpus_dir,
            options.output_dir,
            options.update_limit,
            options.small_model,
        )
        plan = select_runs(build_run_plan(context.design, PlanScope(options.scope)), options.runs)
        ledger = RunLedger(options.output_dir / LEDGER_FILE_NAME)
        pending = ledger.pending_runs(plan)
        upcoming = pending[: options.max_runs] if options.max_runs is not None else pending
        print(f"{len(pending)} of {len(plan)} selected runs are pending.", flush=True)
        for run in upcoming:
            print(f"  {run.run_id}", flush=True)

        with StudyLock(options.output_dir / LOCK_FILE_NAME):
            executed = execute_pending_runs(context, plan, options.max_runs)
        remaining = len(ledger.pending_runs(plan))
        print(f"{executed} runs executed; {remaining} selected runs still pending.", flush=True)
    except Exception:
        traceback.print_exc()
        print(FAILURE_MARKER, flush=True)
        raise
    print(COMPLETION_MARKER, flush=True)


if __name__ == "__main__":
    main()
