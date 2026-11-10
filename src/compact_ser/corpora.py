"""Corpus manifests for IEMOCAP, RAVDESS, and Berlin Emo-DB, with sealed test partitions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from compact_ser.study_design import SessionFold

IEMOCAP_EMOTIONS = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
}
RAVDESS_EMOTIONS = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}
RAVDESS_STATEMENTS = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}
EMODB_EMOTIONS = {"W": "angry", "F": "happy", "T": "sad", "N": "neutral"}
# Published four-class IEMOCAP counts: Gao et al., Interspeech 2023, section 3.1.
IEMOCAP_FOUR_CLASS_COUNTS = {"happy": 1636, "sad": 1084, "angry": 1103, "neutral": 1708}

_EVALUATION_LINE = re.compile(r"^\[[\d.]+ - [\d.]+\]\s+(?P<utterance>Ses\S+)\s+(?P<label>\w+)\s+\[")
_TRANSCRIPT_LINE = re.compile(r"^(?P<utterance>Ses\S+)\s+\[[\d.\-]+\]:\s*(?P<text>.*)$")


class Partition(StrEnum):
    """Partitions of a cross-validation fold."""

    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class SealedPartitionError(RuntimeError):
    """Raised when held-out test data is requested without being unsealed."""


@dataclass(frozen=True)
class Utterance:
    """One labelled utterance in a corpus.

    Attributes:
        utterance_id: Identifier from the corpus.
        corpus: Name of the corpus.
        speaker: Speaker identifier, unique within the corpus.
        emotion: Harmonized emotion class.
        source_label: Emotion label used by the corpus.
        interaction: How the speech was elicited: `improvised`, `scripted`, or `acted`.
        transcript: Spoken text, or an empty string if the corpus has none in this manifest.
        audio_path: Path to the audio, relative to the corpus folder.
        session: Recording session, for corpora that have sessions.
    """

    utterance_id: str
    corpus: str
    speaker: str
    emotion: str
    source_label: str
    interaction: str
    transcript: str
    audio_path: str
    session: int | None = None


def _visible_text_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.txt") if not path.name.startswith("."))


def _read_iemocap_transcripts(folder: Path) -> dict[str, str]:
    transcripts = {}
    for path in _visible_text_files(folder):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _TRANSCRIPT_LINE.match(line.strip())
            if match:
                transcripts[match["utterance"]] = match["text"].strip()
    return transcripts


def read_iemocap(corpus_root: Path, sessions: Iterable[int]) -> list[Utterance]:
    """Read the four-class IEMOCAP utterances from the release folder.

    Args:
        corpus_root: Folder containing `Session1` to `Session5`.
        sessions: Session numbers to read.

    Returns:
        Utterances with an included emotion label, sorted by identifier.

    Raises:
        FileNotFoundError: If a session folder is missing or has no evaluation files.
        ValueError: If an included utterance has no transcript.
    """
    utterances = []
    for session in sessions:
        dialog_root = corpus_root / f"Session{session}" / "dialog"
        evaluation_files = _visible_text_files(dialog_root / "EmoEvaluation")
        if not evaluation_files:
            raise FileNotFoundError(f"No IEMOCAP evaluation files found in {dialog_root}.")
        transcripts = _read_iemocap_transcripts(dialog_root / "transcriptions")
        for evaluation_file in evaluation_files:
            dialog = evaluation_file.stem
            content = evaluation_file.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines():
                match = _EVALUATION_LINE.match(line.strip())
                if not match or match["label"] not in IEMOCAP_EMOTIONS:
                    continue
                utterance_id = match["utterance"]
                if utterance_id not in transcripts:
                    raise ValueError(f"{utterance_id} has no transcript.")
                speaker_gender = utterance_id.rsplit("_", maxsplit=1)[-1][0]
                utterances.append(
                    Utterance(
                        utterance_id=utterance_id,
                        corpus="iemocap",
                        speaker=f"{utterance_id[:5]}_{speaker_gender}",
                        emotion=IEMOCAP_EMOTIONS[match["label"]],
                        source_label=match["label"],
                        interaction="improvised" if "impro" in dialog else "scripted",
                        transcript=transcripts[utterance_id],
                        audio_path=(f"Session{session}/sentences/wav/{dialog}/{utterance_id}.wav"),
                        session=session,
                    )
                )
    return sorted(utterances, key=lambda utterance: utterance.utterance_id)


def read_ravdess(corpus_root: Path) -> list[Utterance]:
    """Read four-class audio-only speech from RAVDESS using its file naming scheme.

    Args:
        corpus_root: Folder containing the RAVDESS audio-only speech files.

    Returns:
        Utterances with an included emotion label, sorted by identifier.
    """
    utterances = []
    for path in corpus_root.rglob("*.wav"):
        modality, channel, emotion, _, statement, _, actor = path.stem.split("-")
        if (modality, channel) != ("03", "01") or emotion not in RAVDESS_EMOTIONS:
            continue
        utterances.append(
            Utterance(
                utterance_id=path.stem,
                corpus="ravdess",
                speaker=f"Actor_{actor}",
                emotion=RAVDESS_EMOTIONS[emotion],
                source_label=emotion,
                interaction="acted",
                transcript=RAVDESS_STATEMENTS[statement],
                audio_path=path.relative_to(corpus_root).as_posix(),
            )
        )
    return sorted(utterances, key=lambda utterance: utterance.utterance_id)


def read_emodb(corpus_root: Path) -> list[Utterance]:
    """Read four-class utterances from Berlin Emo-DB using its file naming scheme.

    Args:
        corpus_root: Folder containing the Emo-DB audio files.

    Returns:
        Utterances with an included emotion label, sorted by identifier.
    """
    utterances = []
    for path in corpus_root.rglob("*.wav"):
        emotion = path.stem[5]
        if emotion not in EMODB_EMOTIONS:
            continue
        utterances.append(
            Utterance(
                utterance_id=path.stem,
                corpus="emodb",
                speaker=path.stem[:2],
                emotion=EMODB_EMOTIONS[emotion],
                source_label=emotion,
                interaction="acted",
                transcript="",
                audio_path=path.relative_to(corpus_root).as_posix(),
            )
        )
    return sorted(utterances, key=lambda utterance: utterance.utterance_id)


def write_manifest(utterances: Iterable[Utterance], path: Path) -> None:
    """Write utterances to a JSON Lines manifest.

    Args:
        utterances: Utterances to write.
        path: Output file. Manifests hold licensed labels and text, so keep them out of git.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for utterance in utterances:
            stream.write(json.dumps(asdict(utterance), sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[Utterance]:
    """Read utterances from a JSON Lines manifest.

    Args:
        path: Manifest file.

    Returns:
        The utterances in file order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Utterance(**json.loads(line)) for line in lines if line.strip()]


def select_partition(
    utterances: Sequence[Utterance],
    fold: SessionFold,
    partition: Partition,
    *,
    unseal_test: bool = False,
) -> list[Utterance]:
    """Return the utterances that belong to one partition of a fold.

    Args:
        utterances: Utterances from a corpus with sessions.
        fold: The cross-validation fold.
        partition: The partition to select.
        unseal_test: Must be true to read the test partition. Only final evaluation and the
            transcript audit set it.

    Returns:
        The selected utterances.

    Raises:
        SealedPartitionError: If the test partition is requested while sealed.
    """
    if partition is Partition.TEST and not unseal_test:
        raise SealedPartitionError(
            f"Session {fold.test_session} is sealed for testing. Only final evaluation may read it."
        )
    sessions = {
        Partition.TRAINING: set(fold.training_sessions),
        Partition.VALIDATION: {fold.validation_session},
        Partition.TEST: {fold.test_session},
    }[partition]
    return [utterance for utterance in utterances if utterance.session in sessions]
