"""Records of the software and hardware used in each session."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

BUNDLE_CHECKSUM_VARIABLE = "COMPACT_SER_BUNDLE_SHA256"


def describe_runtime() -> dict[str, Any]:
    """Return the main software and hardware details of this session."""
    gpu_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if gpu_available else None,
        "gpu_memory_gib": (
            torch.cuda.get_device_properties(0).total_memory / 2**30 if gpu_available else None
        ),
        "code_bundle_sha256": os.environ.get(BUNDLE_CHECKSUM_VARIABLE),
    }


def _command_output(*command: str) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return f"{command[0]} is not available.\n"
    return completed.stdout


def save_runtime_record(output_dir: Path) -> Path:
    """Save the session details, GPU status, and installed packages.

    Args:
        output_dir: Folder that receives the timestamped record.

    Returns:
        The folder containing the record.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    record_dir = output_dir / "environment" / timestamp
    record_dir.mkdir(parents=True, exist_ok=True)
    session = json.dumps(describe_runtime(), indent=2, sort_keys=True) + "\n"
    (record_dir / "session.json").write_text(session, encoding="utf-8")
    (record_dir / "nvidia_smi.txt").write_text(_command_output("nvidia-smi"), encoding="utf-8")
    packages = _command_output(sys.executable, "-m", "pip", "freeze")
    (record_dir / "pip_freeze.txt").write_text(packages, encoding="utf-8")
    return record_dir
