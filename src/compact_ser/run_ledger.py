"""Append-only record of run progress, used to resume work after a disconnected session."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from compact_ser.study_design import RunSpec


class RunStatus(StrEnum):
    """Recorded states of a training run."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class RunLedger:
    """Progress log stored as one JSON object per line.

    A session can end while a line is being written. Such an unfinished final line is ignored
    when the ledger is read, and is repaired before the next line is appended, so that it never
    ends up in the middle of the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _repair_final_line(self) -> None:
        """Finish or remove a final line that a stopped session left without a line break.

        A final line that is a complete entry is kept and given its line break. A partial entry
        is removed: its write never finished, so the change it recorded did not take effect.
        """
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except ValueError:
            with self.path.open("r+b") as stream:
                stream.truncate(start)
                stream.flush()
                os.fsync(stream.fileno())
            print(f"Removed an unfinished final line from {self.path}.", flush=True)
            return
        with self.path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record(self, run_id: str, status: RunStatus, **details: Any) -> None:
        """Append a status change and force it to storage.

        Args:
            run_id: Identifier of the run.
            status: The new status.
            **details: Additional JSON-serializable fields to store.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_final_line()
        entry = {
            "run_id": run_id,
            "status": status.value,
            "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **details,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def latest_statuses(self) -> dict[str, RunStatus]:
        """Return the most recent status of each recorded run.

        Raises:
            ValueError: If a line before the end of the file cannot be parsed.
        """
        if not self.path.exists():
            return {}
        statuses: dict[str, RunStatus] = {}
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                if number == len(lines):
                    break
                raise ValueError(f"{self.path} has a damaged line at {number}.") from error
            statuses[entry["run_id"]] = RunStatus(entry["status"])
        return statuses

    def pending_runs(self, plan: Iterable[RunSpec]) -> list[RunSpec]:
        """Return planned runs that have not completed, keeping their order.

        Args:
            plan: Run specifications in execution order.

        Returns:
            The runs that still need to be executed.
        """
        statuses = self.latest_statuses()
        return [run for run in plan if statuses.get(run.run_id) is not RunStatus.COMPLETED]
