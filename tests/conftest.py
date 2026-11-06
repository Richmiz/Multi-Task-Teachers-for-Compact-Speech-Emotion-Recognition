"""Shared fixtures, including a small synthetic corpus in the IEMOCAP folder layout."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

LABELS = ("neu", "hap", "exc", "sad", "ang", "fru", "neu", "sad", "ang", "hap")
SAMPLE_RATE = 16_000


def write_dialog(root: Path, session: int, dialog: str, rng: np.random.Generator) -> None:
    dialog_root = root / f"Session{session}" / "dialog"
    audio_root = root / f"Session{session}" / "sentences" / "wav" / dialog
    (dialog_root / "EmoEvaluation").mkdir(parents=True, exist_ok=True)
    (dialog_root / "transcriptions").mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    evaluations = ["% Header line that the parser must skip", ""]
    transcripts = [f"{dialog}_MXX [000.0000-001.0000]: [LAUGHTER]"]
    for index, label in enumerate(LABELS):
        gender = "F" if index % 2 == 0 else "M"
        utterance = f"{dialog}_{gender}{index:03d}"
        evaluations.append(
            f"[{index}.0000 - {index + 1}.0000]\t{utterance}\t{label}\t[2.5000, 2.5000, 2.5000]"
        )
        evaluations.append("C-E1:\tNeutral;\t()")
        words = f"session {session} line {index}" if "impro" in dialog else f"shared line {index}"
        transcripts.append(f"{utterance} [{index:03d}.0000-{index + 1:03d}.0000]: {words}.")
        samples = rng.normal(0.0, 0.1, SAMPLE_RATE // 2).astype(np.float32)
        soundfile.write(audio_root / f"{utterance}.wav", samples, SAMPLE_RATE)
    evaluation_text = "\n".join(evaluations) + "\n"
    (dialog_root / "EmoEvaluation" / f"{dialog}.txt").write_text(evaluation_text, encoding="utf-8")
    transcript_text = "\n".join(transcripts) + "\n"
    (dialog_root / "transcriptions" / f"{dialog}.txt").write_text(transcript_text, encoding="utf-8")


@pytest.fixture(scope="session")
def iemocap_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("iemocap")
    rng = np.random.default_rng(0)
    for session in range(1, 6):
        write_dialog(root, session, f"Ses0{session}F_impro01", rng)
        write_dialog(root, session, f"Ses0{session}M_script01_1", rng)
    return root
