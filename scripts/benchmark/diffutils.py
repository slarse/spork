"""Diff utility functions."""
import dataclasses
import difflib
import pathlib


@dataclasses.dataclass
class DiffStats:
    ratio: float
    diff_size: int


def compute_character_diff(
    base_file: pathlib.Path, dest_file: pathlib.Path
) -> DiffStats:
    """Compute a character-based diff between the two provided files."""
    matcher = difflib.SequenceMatcher(
        a=base_file.read_text(encoding="utf8"),
        b=dest_file.read_text(encoding="utf8"),
        autojunk=False,
    )
    ratio = matcher.ratio()
    diff_size = _compute_diff_size(matcher)
    return DiffStats(ratio, diff_size)


def _compute_diff_size(matcher: difflib.SequenceMatcher) -> int:
    """Compute the diff size represented by the matcher.

    Note that the available operations are 'replace', 'delete', 'insert' and
    'equal'.  Only 'equal' does not contribute to the diff size.

    For all other tags, [base_start, base_end) is the range of characters from
    the base revision that should be replaced with [dest_start, dest_end) from
    the destination revision. For 'delete', the destination range is empty, and
    for 'insert', the base range is empty. For 'replace' both ranges are
    non-empty. The diff size is therefore simply the sum the sizes of both base
    and destination ranges, where the base range represents deletions and the
    destination range represents insertions ('replace' is just a deletion and
    insertion modeled as one operation).
    """
    return sum(
        [
            (base_end - base_start) + (dest_end - dest_start)
            for tag, base_start, base_end, dest_start, dest_end in matcher.get_opcodes()
            if tag != "equal"
        ]
    )
