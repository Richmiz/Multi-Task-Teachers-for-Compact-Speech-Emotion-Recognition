"""Transcript normalization, CTC tokenization, edit distance, and text-overlap checks."""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Sequence

BLANK_TOKEN = 0
UNKNOWN_TOKEN = 1

_NON_SPEECH_EVENT = re.compile(r"\[[^\]]*\]|<[^>]*>")
_UNSUPPORTED_CHARACTER = re.compile(r"[^A-Z' ]")
_WHITESPACE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """Convert a transcript to upper-case letters, apostrophes, and single spaces.

    Non-speech events in square or angle brackets are removed.

    Args:
        text: The original transcript.

    Returns:
        The normalized transcript.
    """
    without_events = _NON_SPEECH_EVENT.sub(" ", text)
    letters = _UNSUPPORTED_CHARACTER.sub(" ", without_events.upper())
    return _WHITESPACE.sub(" ", letters).strip()


class TranscriptTokenizer:
    """Character tokenizer for CTC, with a vocabulary learned from training transcripts."""

    def __init__(self, characters: Iterable[str]) -> None:
        self.characters = sorted(set(characters))
        offset = UNKNOWN_TOKEN + 1
        self._tokens = {character: offset + i for i, character in enumerate(self.characters)}
        self._characters = {token: character for character, token in self._tokens.items()}

    @classmethod
    def fit(cls, transcripts: Iterable[str]) -> TranscriptTokenizer:
        """Create a tokenizer from the characters in normalized training transcripts.

        Args:
            transcripts: Original training transcripts.

        Returns:
            The fitted tokenizer.
        """
        characters: set[str] = set()
        for transcript in transcripts:
            characters.update(normalize_transcript(transcript))
        return cls(characters)

    @property
    def vocabulary_size(self) -> int:
        """Number of tokens, including the blank and unknown tokens."""
        return len(self.characters) + UNKNOWN_TOKEN + 1

    def encode(self, transcript: str) -> list[int]:
        """Convert a transcript to token indices.

        Args:
            transcript: The original transcript.

        Returns:
            One token per normalized character. Unseen characters use the unknown token.
        """
        normalized = normalize_transcript(transcript)
        return [self._tokens.get(character, UNKNOWN_TOKEN) for character in normalized]

    def decode_ctc(self, tokens: Iterable[int]) -> str:
        """Collapse repeated tokens, remove blanks, and return the text.

        Args:
            tokens: Most likely token for each frame.

        Returns:
            The decoded transcript.
        """
        characters = []
        previous = None
        for token in tokens:
            if token != previous and token in self._characters:
                characters.append(self._characters[token])
            previous = token
        return "".join(characters).strip()


def edit_distance(first: Sequence, second: Sequence, limit: int | None = None) -> int:
    """Compute the Levenshtein distance between two sequences.

    Args:
        first: First sequence.
        second: Second sequence.
        limit: If given, stop early and return `limit + 1` once the distance must exceed it.

    Returns:
        The edit distance, or `limit + 1` if it exceeds the limit.
    """
    if limit is not None and abs(len(first) - len(second)) > limit:
        return limit + 1
    previous = list(range(len(second) + 1))
    for row, first_item in enumerate(first, start=1):
        current = [row]
        for column, second_item in enumerate(second, start=1):
            substitution = previous[column - 1] + (first_item != second_item)
            current.append(min(previous[column] + 1, current[column - 1] + 1, substitution))
        if limit is not None and min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute the word error rate across a set of transcripts.

    Args:
        references: Normalized reference transcripts.
        hypotheses: Decoded transcripts, in the same order.

    Returns:
        Total word edits divided by the total number of reference words.
    """
    edits = 0
    words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = reference.split()
        edits += edit_distance(reference_words, hypothesis.split())
        words += len(reference_words)
    return edits / max(words, 1)


def find_seen_transcripts(
    test_transcripts: Iterable[str],
    training_transcripts: Iterable[str],
    max_distance_ratio: float,
) -> list[bool]:
    """Mark test transcripts that also occur, exactly or nearly, in the training transcripts.

    Args:
        test_transcripts: Original transcripts to check.
        training_transcripts: Original transcripts from the training sessions.
        max_distance_ratio: Largest edit distance for a near match, as a share of the longer text.

    Returns:
        One flag for each test transcript.
    """
    known = {normalize_transcript(transcript) for transcript in training_transcripts}
    by_length = sorted(known, key=len)
    lengths = [len(text) for text in by_length]

    def is_seen(text: str) -> bool:
        if text in known:
            return True
        longest = int(len(text) / (1 - max_distance_ratio)) if max_distance_ratio < 1 else None
        shortest = int(len(text) * (1 - max_distance_ratio))
        start = bisect.bisect_left(lengths, shortest)
        stop = len(lengths) if longest is None else bisect.bisect_right(lengths, longest)
        for candidate in by_length[start:stop]:
            limit = int(max_distance_ratio * max(len(text), len(candidate)))
            if edit_distance(text, candidate, limit) <= limit:
                return True
        return False

    return [is_seen(normalize_transcript(transcript)) for transcript in test_transcripts]
