"""Run this package's jobs on a Google Colab runtime from one fixed notebook.

`notebooks/colab_runner.ipynb` fetches the repository at a chosen commit and calls these
functions. A job runs as a background process, so stopping a notebook cell does not stop it, and
its console output is saved to Google Drive with the commit it ran. The VS Code Colab kernel does
not show a background process's output, so `follow_job` relays the saved log instead.

This module uses only the standard library, so the notebook can import it before the job's
dependencies are checked.
"""

from __future__ import annotations

import getpass
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive"
LOCAL_DATA_DIR = Path("/content/compact-ser-data")
CODE_COMMIT_VARIABLE = "COMPACT_SER_CODE_COMMIT"
REQUIRED_PACKAGES = ("scipy", "soundfile", "torch", "transformers")
LOG_FILE_NAME = "job.log"
LOG_HEADER_PREFIX = "--- "
IEMOCAP_STORE_DIR = "{output_root}/datasets/iemocap"


@dataclass(frozen=True)
class ColabJob:
    """A package module that runs as a background job.

    Attributes:
        module: Module run with `python -m`.
        folder: Folder in the project's Drive folder that receives the job's outputs.
        completion_marker: Line the job prints when it completes.
        failure_marker: Line the job prints when it fails.
        arguments: Arguments placed before the user's options. `{output_root}`, `{drive_root}`,
            and `{local_data}` stand for the project's Drive folder, the root of My Drive, and
            the runtime's local data folder.
        requires_gpu: Whether the job needs a GPU runtime.
        asks_for_password: Whether the job reads a password from standard input.
    """

    module: str
    folder: str
    completion_marker: str
    failure_marker: str
    arguments: tuple[str, ...] = ()
    requires_gpu: bool = True
    asks_for_password: bool = False


JOBS = {
    "runtime-check": ColabJob(
        module="compact_ser.runtime_check",
        folder="runtime_check",
        completion_marker="RUNTIME CHECK COMPLETE",
        failure_marker="RUNTIME CHECK FAILED",
    ),
    "iemocap-store": ColabJob(
        module="compact_ser.iemocap_archive",
        folder="iemocap_store",
        completion_marker="IEMOCAP STORE COMPLETE",
        failure_marker="IEMOCAP STORE FAILED",
        arguments=(
            "store",
            "--work-dir",
            "/content/iemocap-download",
            "--store-dir",
            IEMOCAP_STORE_DIR,
        ),
        requires_gpu=False,
        asks_for_password=True,
    ),
    "iemocap-check": ColabJob(
        module="compact_ser.iemocap_archive",
        folder="iemocap_check",
        completion_marker="IEMOCAP CHECK COMPLETE",
        failure_marker="IEMOCAP CHECK FAILED",
        arguments=(
            "check",
            "--store-dir",
            IEMOCAP_STORE_DIR,
            "--search-dir",
            "{drive_root}",
            "--local-dir",
            "{local_data}",
        ),
        requires_gpu=False,
    ),
    "training": ColabJob(
        module="compact_ser.training_job",
        folder="study",
        completion_marker="TRAINING JOB COMPLETE",
        failure_marker="TRAINING JOB FAILED",
        arguments=(
            "--store-dir",
            IEMOCAP_STORE_DIR,
            "--search-dir",
            "{drive_root}",
            "--local-dir",
            "{local_data}",
        ),
    ),
}


@dataclass
class ColabSession:
    """A job prepared on this runtime.

    Attributes:
        job: The job to run.
        output_root: The project's folder on Google Drive.
        code_root: The repository checkout that the job runs from.
        commit: The commit of that checkout.
        process: The job's process, if it was started by this session.
        log_start: First line of the job's log that belongs to this run.
    """

    job: ColabJob
    output_root: Path
    code_root: Path
    commit: str
    process: subprocess.Popen | None = field(default=None, repr=False)
    log_start: int = 0

    @property
    def job_root(self) -> Path:
        """Folder that receives the job's outputs."""
        return self.output_root / self.job.folder

    @property
    def log_path(self) -> Path:
        """The job's console output, appended across runs."""
        return self.job_root / LOG_FILE_NAME

    @property
    def pid_path(self) -> Path:
        """Local file holding the process ID of a running job."""
        return Path(tempfile.gettempdir()) / f"compact_ser_{self.job.folder}.pid"


def find_job(name: str) -> ColabJob:
    """Return a job by name.

    Raises:
        ValueError: If no job has that name.
    """
    if name not in JOBS:
        raise ValueError(f"Unknown job {name!r}. Choose one of: {', '.join(sorted(JOBS))}.")
    return JOBS[name]


def mount_drive(drive_folder: str) -> Path:
    """Mount Google Drive and return the project's folder on it, creating the folder if needed.

    Args:
        drive_folder: The project's folder, relative to My Drive.

    Raises:
        RuntimeError: If Drive cannot be mounted from this kernel.
    """
    from google.colab import drive

    if not os.path.ismount(DRIVE_MOUNT):
        try:
            drive.mount(str(DRIVE_MOUNT))
        except Exception as error:
            raise RuntimeError(
                'Google Drive is not mounted. In VS Code, run "Colab: Mount Google Drive to '
                f'Server...", then run this cell again. ({error})'
            ) from error
    try:
        os.listdir(DRIVE_ROOT)
    except OSError:
        # A dropped connection ("Transport endpoint is not connected") needs a fresh mount.
        drive.mount(str(DRIVE_MOUNT), force_remount=True)
    output_root = DRIVE_ROOT / drive_folder
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def install_missing_packages(packages: Sequence[str] = REQUIRED_PACKAGES) -> None:
    """Install the packages that this runtime does not already provide."""
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def prepare_session(job_name: str, drive_folder: str, code_root: Path, commit: str) -> ColabSession:
    """Mount Drive, check the runtime, and prepare a job.

    Args:
        job_name: Name of a job in `JOBS`.
        drive_folder: The project's folder, relative to My Drive.
        code_root: The repository checkout that the job runs from.
        commit: The commit of that checkout.

    Returns:
        The prepared session.

    Raises:
        RuntimeError: If the job needs a GPU and the runtime has none.
    """
    job = find_job(job_name)
    output_root = mount_drive(drive_folder)
    install_missing_packages()
    if job.requires_gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"The {job_name} job needs a GPU. Connect a Colab GPU server.")
        memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {torch.cuda.get_device_name(0)} ({memory:.0f} GiB), {os.cpu_count()} CPUs.")
    else:
        print(f"No GPU is needed. {os.cpu_count()} CPUs.")
    session = ColabSession(job, output_root, Path(code_root), commit)
    session.job_root.mkdir(parents=True, exist_ok=True)
    print(f"Outputs are saved in {session.job_root}")
    return session


def job_command(session: ColabSession, options: Sequence[str] = ()) -> list[str]:
    """Return the command that runs the job.

    Args:
        session: The prepared session.
        options: Options added after the job's own arguments.

    Returns:
        The command, with the job's output folder as `--output-dir`.
    """
    places = {
        "output_root": session.output_root,
        "drive_root": DRIVE_ROOT,
        "local_data": LOCAL_DATA_DIR,
    }
    arguments = [argument.format(**places) for argument in session.job.arguments]
    return [
        sys.executable,
        "-u",
        "-m",
        session.job.module,
        *arguments,
        "--output-dir",
        str(session.job_root),
        *options,
    ]


def is_running(session: ColabSession) -> bool:
    """Return whether the session's job is running, including one started by an earlier kernel."""
    if session.process is not None:
        return session.process.poll() is None
    if not session.pid_path.exists():
        return False
    status = Path(f"/proc/{session.pid_path.read_text().strip()}/stat")
    return status.exists() and status.read_text().rsplit(") ", 1)[-1][0] != "Z"


def _read_log(session: ColabSession) -> list[str]:
    if not session.log_path.exists():
        return []
    return session.log_path.read_text(encoding="utf-8", errors="replace").splitlines()


def start_job(
    session: ColabSession, options: Sequence[str] = (), password: str | None = None
) -> None:
    """Start the job in the background, unless it is already running.

    Args:
        session: The prepared session.
        options: Options added after the job's own arguments.
        password: Password for a job that asks for one. If it is not given, it is requested
            without being shown. It reaches the job through a pipe, so it is never saved.
    """
    lines = _read_log(session)
    if is_running(session):
        headers = [i for i, line in enumerate(lines) if line.startswith(LOG_HEADER_PREFIX)]
        session.log_start = headers[-1] if headers else 0
        print(f"The job is already running, PID {session.pid_path.read_text().strip()}.")
        return
    session.log_start = len(lines)
    if session.job.asks_for_password and password is None:
        password = getpass.getpass("Download password: ")
    command = job_command(session, options)
    environment = {
        **os.environ,
        "PYTHONPATH": str(session.code_root / "src"),
        CODE_COMMIT_VARIABLE: session.commit,
    }
    session.job_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    with session.log_path.open("a", encoding="utf-8") as log:
        log.write(f"{LOG_HEADER_PREFIX}{started} {session.job.module} at commit {session.commit}\n")
        log.flush()
        session.process = subprocess.Popen(
            command,
            cwd=session.code_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    session.process.stdin.write((password or "") + "\n")
    session.process.stdin.close()
    session.pid_path.write_text(str(session.process.pid))
    print(f"Job started, PID {session.process.pid}. Its log is saved to {session.log_path}.")


def _resource_usage() -> str:
    if not shutil.which("nvidia-smi"):
        return "running"
    usage = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"GPU {usage}"


def follow_job(session: ColabSession, interval: float = 30.0) -> bool:
    """Show the job's log as it grows, until the job completes, fails, or stops.

    Args:
        session: The session whose job was started.
        interval: Seconds between checks.

    Returns:
        Whether the job printed its completion marker.
    """
    shown = 0
    while True:
        # Checking the process first means that, once it has stopped, its whole log is read.
        running = is_running(session)
        lines = _read_log(session)[session.log_start :]
        for line in lines[shown:]:
            print(line, flush=True)
        shown = len(lines)
        completed = any(session.job.completion_marker in line for line in lines)
        failed = any(session.job.failure_marker in line for line in lines)
        if completed or failed or not running:
            break
        print(f"  ... {time.strftime('%H:%M:%S')} {_resource_usage()}", flush=True)
        time.sleep(interval)
    print("Job completed." if completed else "Job stopped before completing. See the log above.")
    return completed


def shut_down_runtime(completed: bool, delay: float = 60.0) -> None:
    """Release the Colab runtime after a completed job, so that it stops billing.

    Args:
        completed: Whether the job completed.
        delay: Seconds to wait first, so the last Drive writes finish.

    Raises:
        RuntimeError: If the job did not complete. The runtime is then kept for inspection.
    """
    if not completed:
        raise RuntimeError(
            "The job did not complete. Check the log, then disconnect the runtime yourself."
        )
    print(f"Results are on Google Drive. Shutting the runtime down in {delay:.0f} seconds...")
    time.sleep(delay)
    from google.colab import runtime

    runtime.unassign()
