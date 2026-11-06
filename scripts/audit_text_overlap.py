"""Report how many test utterances repeat a transcript from their fold's training sessions.

The audit reads test transcripts only. It uses no models, predictions, or test audio. The default
output contains aggregate counts, while utterance flags are written beside the manifest.

    python scripts/audit_text_overlap.py --manifest data/manifests/iemocap.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compact_ser.corpora import Partition, read_manifest, select_partition
from compact_ser.study_design import load_study_design, session_folds
from compact_ser.text_processing import find_seen_transcripts


def main() -> None:
    """Parse command-line arguments and write the overlap report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-distance-ratio", type=float, default=0.1)
    options = parser.parse_args()

    utterances = read_manifest(options.manifest)
    flags: dict[str, bool] = {}
    report = {"max_distance_ratio": options.max_distance_ratio, "folds": []}
    for fold in session_folds(load_study_design().sessions):
        test = select_partition(utterances, fold, Partition.TEST, unseal_test=True)
        training = select_partition(utterances, fold, Partition.TRAINING)
        seen = find_seen_transcripts(
            (utterance.transcript for utterance in test),
            (utterance.transcript for utterance in training),
            options.max_distance_ratio,
        )
        flags.update(
            {utterance.utterance_id: is_seen for utterance, is_seen in zip(test, seen, strict=True)}
        )
        counts = Counter(
            (utterance.interaction, utterance.emotion, "seen" if is_seen else "unseen")
            for utterance, is_seen in zip(test, seen, strict=True)
        )
        report["folds"].append(
            {
                "test_session": fold.test_session,
                "utterances": len(test),
                "text_seen": sum(seen),
                "counts": [
                    {"interaction": interaction, "emotion": emotion, "text": text, "count": count}
                    for (interaction, emotion, text), count in sorted(counts.items())
                ],
            }
        )
        print(
            f"Session {fold.test_session}: {sum(seen)} of {len(test)} transcripts seen in training"
        )

    output_dir = options.manifest.parent
    report_path = output_dir / f"{options.manifest.stem}_text_overlap.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    flags_path = output_dir / f"{options.manifest.stem}_text_seen.json"
    flags_path.write_text(json.dumps(flags, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
