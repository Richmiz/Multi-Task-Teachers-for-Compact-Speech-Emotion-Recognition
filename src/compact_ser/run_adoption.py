"""Adopt completed teacher runs from an earlier study folder into the current study.

The validation pilot trained emotion-only teachers with the study's seeds, and multi-task teachers
with several auxiliary weights. A pilot teacher is identical to a study teacher when its training
settings, seed, and auxiliary weight match the current design, so it is moved into the study
instead of being trained again. Each adoption is recorded in the result and in the run ledger.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from compact_ser.run_ledger import RunLedger, RunStatus
from compact_ser.study_design import RunKind, RunSpec, StudyDesign, TeacherObjective
from compact_ser.study_runner import (
    LEDGER_FILE_NAME,
    LOCK_FILE_NAME,
    MODEL_FILE_NAME,
    RESULT_FILE_NAME,
    SETTINGS_FILE_NAME,
)

PILOT_TEACHER_ID = re.compile(
    r"^teacher_tuning__(?P<objective>[a-z_]+)__session(?P<session>\d+)"
    r"__candidate\d+__seed(?P<seed>\d+)$"
)
# Settings that do not affect how a teacher is trained.
STUDENT_ONLY_SETTINGS = {"student_updates"}
LOCK_STALE_SECONDS = 600.0

Logger = Callable[[str], None]


def _teacher_settings(record: dict[str, Any]) -> dict[str, Any]:
    training = {k: v for k, v in record["training"].items() if k not in STUDENT_ONLY_SETTINGS}
    shared = ("checkpoint_name", "checkpoint_revision", "sessions", "emotions")
    return {**{name: record[name] for name in shared}, "training": training}


def _check_source(source_root: Path, design: StudyDesign) -> None:
    lock = source_root / LOCK_FILE_NAME
    if lock.exists():
        refreshed = json.loads(lock.read_text(encoding="utf-8"))["refreshed_at"]
        if time.time() - float(refreshed) < LOCK_STALE_SECONDS:
            raise RuntimeError(
                f"A runtime is still training in {source_root}. Wait for it to stop."
            )
    saved = json.loads((source_root / SETTINGS_FILE_NAME).read_text(encoding="utf-8"))
    current = json.loads(json.dumps(asdict(design)))
    if _teacher_settings(saved) != _teacher_settings(current):
        raise ValueError(f"The teachers in {source_root} were trained with other settings.")


def adoptable_run_id(old_run_id: str, result: dict[str, Any], design: StudyDesign) -> str | None:
    """Return the study run that a completed pilot teacher matches, if any.

    Args:
        old_run_id: Identifier of the pilot run.
        result: The pilot run's saved result.
        design: The current study settings.

    Returns:
        The identifier of the matching study teacher, or `None`.
    """
    match = PILOT_TEACHER_ID.match(old_run_id)
    if match is None or int(match["seed"]) not in design.seeds:
        return None
    objective = TeacherObjective(match["objective"])
    if objective is TeacherObjective.ASR_THEN_SER:
        matches = result.get("asr_update_share") == design.asr_update_share
    elif objective is TeacherObjective.SER_ONLY:
        matches = True
    else:
        matches = result.get("auxiliary_loss_scale") == design.auxiliary_loss_scale
    if not matches:
        return None
    spec = RunSpec(RunKind.TEACHER, int(match["session"]), objective, seed=int(match["seed"]))
    return spec.run_id


def adopt_runs(
    source_root: Path, output_root: Path, design: StudyDesign, log: Logger = print
) -> list[str]:
    """Move completed pilot teachers that match the current design into the study folder.

    Args:
        source_root: The earlier study folder.
        output_root: The current study folder.
        design: The current study settings.
        log: Receives progress messages.

    Returns:
        The identifiers of the adopted runs in the current study.

    Raises:
        RuntimeError: If a runtime is still training in the source folder.
        ValueError: If the source teachers were trained with other settings.
    """
    if not (source_root / "runs").is_dir():
        log(f"No runs to adopt in {source_root}.")
        return []
    _check_source(source_root, design)
    completed = RunLedger(source_root / LEDGER_FILE_NAME).latest_statuses()
    ledger = RunLedger(output_root / LEDGER_FILE_NAME)
    adopted = []
    for folder in sorted((source_root / "runs").iterdir()):
        if completed.get(folder.name) is not RunStatus.COMPLETED:
            continue
        if not ((folder / RESULT_FILE_NAME).is_file() and (folder / MODEL_FILE_NAME).is_file()):
            continue
        result = json.loads((folder / RESULT_FILE_NAME).read_text(encoding="utf-8"))
        run_id = adoptable_run_id(folder.name, result, design)
        target = output_root / "runs" / str(run_id)
        if run_id is None or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(target))
        result.update(run_id=run_id, adopted_from=folder.name)
        (target / RESULT_FILE_NAME).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        ledger.record(
            run_id,
            RunStatus.COMPLETED,
            wall_seconds=result["wall_seconds"],
            adopted_from=str(source_root / "runs" / folder.name),
        )
        adopted.append(run_id)
        log(f"Adopted {folder.name} as {run_id}.")
    log(f"{len(adopted)} completed teachers adopted from {source_root}.")
    return adopted
