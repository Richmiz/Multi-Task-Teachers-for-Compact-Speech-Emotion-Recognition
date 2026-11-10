"""Build corpus manifests from local dataset folders.

Manifests contain licensed labels and transcripts, so write them under `data/`, which git ignores.

    python scripts/build_manifests.py --iemocap-root data/iemocap
    python scripts/build_manifests.py --ravdess-root data/ravdess --emodb-root data/emodb
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from compact_ser.corpora import (
    IEMOCAP_FOUR_CLASS_COUNTS,
    Utterance,
    read_emodb,
    read_iemocap,
    read_ravdess,
    write_manifest,
)
from compact_ser.study_design import load_study_design

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_label_counts(utterances: Sequence[Utterance], path: Path) -> None:
    """Write utterance counts by session, class, corpus label, and interaction type.

    The corpus label shows which happy utterances were labelled excited, for the sensitivity
    analysis that excludes them.

    Args:
        utterances: Utterances in the manifest.
        path: Output file.
    """
    counts = Counter((u.session, u.emotion, u.source_label, u.interaction) for u in utterances)
    speakers = Counter((u.session, u.speaker) for u in utterances)
    report = {
        "counts": [
            {
                "session": session,
                "emotion": emotion,
                "source_label": label,
                "interaction": interaction,
                "count": count,
            }
            for (session, emotion, label, interaction), count in sorted(counts.items())
        ],
        "speakers": [
            {"session": session, "speaker": speaker, "count": count}
            for (session, speaker), count in sorted(speakers.items())
        ],
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Parse command-line arguments and write the requested manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-root", type=Path)
    parser.add_argument("--ravdess-root", type=Path)
    parser.add_argument("--emodb-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "data" / "manifests")
    options = parser.parse_args()

    if options.iemocap_root:
        utterances = read_iemocap(options.iemocap_root, load_study_design().sessions)
        counts = Counter(utterance.emotion for utterance in utterances)
        print(f"IEMOCAP: {len(utterances)} utterances, {dict(counts)}")
        if dict(counts) != IEMOCAP_FOUR_CLASS_COUNTS:
            raise SystemExit(f"IEMOCAP counts differ from {IEMOCAP_FOUR_CLASS_COUNTS}.")
        write_manifest(utterances, options.output_dir / "iemocap.jsonl")
        write_label_counts(utterances, options.output_dir / "iemocap_label_counts.json")
    if options.ravdess_root:
        utterances = read_ravdess(options.ravdess_root)
        print(f"RAVDESS: {len(utterances)} utterances")
        write_manifest(utterances, options.output_dir / "ravdess.jsonl")
    if options.emodb_root:
        utterances = read_emodb(options.emodb_root)
        print(f"Emo-DB: {len(utterances)} utterances")
        write_manifest(utterances, options.output_dir / "emodb.jsonl")


if __name__ == "__main__":
    main()
