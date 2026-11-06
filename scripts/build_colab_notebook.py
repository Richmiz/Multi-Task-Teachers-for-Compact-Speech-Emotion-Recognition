"""Build a self-contained notebook that runs one job on a Google Colab runtime.

The notebook embeds the package and study settings in a checksum-protected bundle, so the runtime
executes exactly the code that built it. Outputs are saved to Google Drive.

    python scripts/build_colab_notebook.py --job runtime-check --tag first-session
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from string import Template

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "notebooks"
DRIVE_FOLDER = "compact-ser"
BUNDLE_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class NotebookJob:
    """A package module that can run as a background Colab job."""

    title: str
    description: str
    module: str
    folder: str
    completion_marker: str
    failure_marker: str


JOBS = {
    "runtime-check": NotebookJob(
        title="Runtime check",
        description=(
            "This job uses no corpus data. It records the environment, loads the pinned "
            "checkpoint, and times training updates on synthetic audio."
        ),
        module="compact_ser.runtime_check",
        folder="runtime_check",
        completion_marker="RUNTIME CHECK COMPLETE",
        failure_marker="RUNTIME CHECK FAILED",
    ),
}

INTRODUCTION_CELL = """\
# Compact SER: $title

$description

**In VS Code:** choose **Select Kernel → Colab → New Colab Server**, select a GPU, then choose \
**Run All**. If Google Drive is not mounted, run **Colab: Mount Google Drive to Server...** from \
the Command Palette, then choose **Run All** again.

Outputs are saved in **My Drive/$drive_folder/$job_folder**. Stopping the progress cell does not \
stop the job. The last cell shuts the runtime down after the job completes. If the job fails, \
disconnect the runtime yourself, because it bills while idle.
"""

BUNDLE_CELL = """\
BUNDLE_SHA256 = "$bundle_sha256"
BUNDLE_BASE64 = "$bundle_base64"
"""

DRIVE_CELL = """\
import os
import pathlib

OUTPUT_ROOT = pathlib.Path("/content/drive/MyDrive/$drive_folder")
if not os.path.ismount("/content/drive"):
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as error:
        raise RuntimeError(
            "Google Drive is not mounted. In VS Code, run "
            '"Colab: Mount Google Drive to Server...", '
            f"then run this cell again. ({error})"
        ) from error
try:
    os.listdir("/content/drive/MyDrive")
except OSError:
    # A dropped connection ("Transport endpoint is not connected") needs a fresh mount.
    from google.colab import drive

    drive.mount("/content/drive", force_remount=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Outputs are saved in {OUTPUT_ROOT}")
"""

INSTALL_CELL = """\
import base64
import hashlib
import importlib.util
import io
import subprocess
import sys
import zipfile

bundle = base64.b64decode(BUNDLE_BASE64)
assert hashlib.sha256(bundle).hexdigest() == BUNDLE_SHA256, "The embedded code bundle is corrupt."
CODE_ROOT = pathlib.Path("/content/compact-ser")
CODE_ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    for name in archive.namelist():
        assert not name.startswith("/") and ".." not in pathlib.PurePosixPath(name).parts
    archive.extractall(CODE_ROOT)
os.chdir(CODE_ROOT)
os.environ["PYTHONPATH"] = str(CODE_ROOT / "src")
os.environ["COMPACT_SER_BUNDLE_SHA256"] = BUNDLE_SHA256
missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

import torch

assert torch.cuda.is_available(), "Select a Colab GPU server before running this notebook."
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"Code verified ({BUNDLE_SHA256[:12]}...).")
print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.0f} GiB), {os.cpu_count()} CPUs.")
"""

START_CELL = """\
JOB_ROOT = OUTPUT_ROOT / "$job_folder"
JOB_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = JOB_ROOT / "job.log"
PID_PATH = CODE_ROOT / "job.pid"


def job_is_running():
    if not PID_PATH.exists():
        return False
    status = pathlib.Path(f"/proc/{PID_PATH.read_text().strip()}/stat")
    return status.exists() and status.read_text().rsplit(") ", 1)[-1][0] != "Z"


LOG_START = len(LOG_PATH.read_text(encoding="utf-8").splitlines()) if LOG_PATH.exists() else 0
if job_is_running():
    print("The job is already running, PID", PID_PATH.read_text().strip())
else:
    command = [sys.executable, "-u", "-m", "$job_module", "--output-dir", str(JOB_ROOT)]
    with LOG_PATH.open("a", encoding="utf-8") as log:
        job = subprocess.Popen(
            command, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    PID_PATH.write_text(str(job.pid))
    print(f"Job started, PID {job.pid}.")
"""

PROGRESS_CELL = """\
import time

COMPLETED = False
shown_lines = 0
while True:
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[LOG_START:]
    for line in lines[shown_lines:]:
        print(line, flush=True)
    shown_lines = len(lines)
    COMPLETED = any("$completion_marker" in line for line in lines)
    if COMPLETED or any("$failure_marker" in line for line in lines) or not job_is_running():
        break
    usage = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"  ... {time.strftime('%H:%M:%S')} GPU {usage}", flush=True)
    time.sleep(30)
print("Job completed." if COMPLETED else "Job stopped before completing. See the log above.")
"""

SHUTDOWN_CELL = """\
assert COMPLETED, "The job did not complete. Check the log, then disconnect the runtime yourself."
print("Results are on Google Drive. Shutting the runtime down in 60 seconds...", flush=True)
time.sleep(60)
from google.colab import runtime

runtime.unassign()
"""


def build_code_bundle(repository_root: Path = REPOSITORY_ROOT) -> bytes:
    """Create a reproducible zip archive of the package and study settings.

    Args:
        repository_root: Root folder of the repository.

    Returns:
        The archive bytes. The same files always produce the same bytes.
    """
    paths = [
        *sorted((repository_root / "src" / "compact_ser").glob("*.py")),
        *sorted((repository_root / "configs").glob("*.toml")),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            info = zipfile.ZipInfo(
                path.relative_to(repository_root).as_posix(), date_time=BUNDLE_TIMESTAMP
            )
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            content = path.read_bytes().replace(b"\r\n", b"\n")
            archive.writestr(info, content, compresslevel=9)
    return buffer.getvalue()


def _create_cell(kind: str, source: str, index: int) -> dict[str, object]:
    cell: dict[str, object] = {
        "cell_type": kind,
        "id": f"cell-{index:02d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_notebook(job: NotebookJob, bundle: bytes) -> dict[str, object]:
    """Assemble the notebook for a job.

    Args:
        job: The job to run.
        bundle: The code bundle to embed.

    Returns:
        The notebook as a JSON-serializable dictionary.
    """
    values = {
        "title": job.title,
        "description": job.description,
        "drive_folder": DRIVE_FOLDER,
        "job_folder": job.folder,
        "job_module": job.module,
        "completion_marker": job.completion_marker,
        "failure_marker": job.failure_marker,
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "bundle_base64": base64.b64encode(bundle).decode("ascii"),
    }
    sources = (
        ("markdown", INTRODUCTION_CELL),
        ("code", BUNDLE_CELL),
        ("code", DRIVE_CELL),
        ("code", INSTALL_CELL),
        ("code", START_CELL),
        ("code", PROGRESS_CELL),
        ("code", SHUTDOWN_CELL),
    )
    cells = [
        _create_cell(kind, Template(source).substitute(values), index)
        for index, (kind, source) in enumerate(sources)
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Parse command-line arguments and write the notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=sorted(JOBS), required=True)
    parser.add_argument("--tag", required=True, help="Label added to the notebook file name.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    options = parser.parse_args()

    job = JOBS[options.job]
    output_path = options.output_dir / f"colab_{job.folder}_{options.tag}.ipynb"
    if output_path.exists():
        raise SystemExit(f"{output_path} exists. Choose a new --tag to keep every run's record.")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(job, build_code_bundle())
    output_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
