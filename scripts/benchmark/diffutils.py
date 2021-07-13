"""Diff utility functions."""
import dataclasses
import difflib
import pathlib
import subprocess
import sys

from typing import List

GIT_DIFF_CMD = tuple(
    "git diff --no-index --numstat --ignore-cr-at-eol --ignore-space-at-eol".split()
)


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


def git_diff_edit_script_size(base_file: pathlib.Path, dest_file: pathlib.Path) -> int:
    """Return the edit script size (insertions + deletions) for the diff
    between the base and destination files, as reported by git-diff. See the
    module constants for which exact arguments are used.

    Args:
        base_file: The base version of the file.
        dest_file: The edited version of the file.
    Returns:
        The size of the edit script.
    """
    cmd = [*GIT_DIFF_CMD, str(base_file), str(dest_file)]
    proc = subprocess.run(cmd, capture_output=True, timeout=10)

    if not proc.stdout:
        return 0

    lines = proc.stdout.decode(sys.getdefaultencoding()).strip().split("\n")
    assert len(lines) == 1

    line = lines[0]
    insertions, deletions, *_ = line.split()
    return int(insertions) + int(deletions)


def git_diff_edit_script(
    base_file: pathlib.Path,
    dest_file: pathlib.Path,
    strip_metadata: bool = False,
) -> List[str]:
    """Return the edit script produced by git diff. Requires that the `git`
    program is on the path.

    Args:
        base_file: The base version of the file.
        dest_file: The edited version of the file.
    Returns:
        The edit script produced by git diff, ignoring carriege returns,
        whitespace and blank lines.
    """
    git_diff = (
        "git diff --ignore-cr-at-eol --ignore-all-space "
        "--ignore-blank-lines --ignore-space-change --no-index -U0"
    ).split()
    cmd = [*git_diff, str(base_file), str(dest_file)]
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=10)

    if proc.returncode == 0:
        # zero exit code means there were no differences
        return []

    output = proc.stdout.decode(sys.getdefaultencoding())

    lines = output.rstrip().split("\n")
    if strip_metadata and lines:
        # first 4 lines are metadata, lines starting with @@ is metadata
        lines = [line for line in lines[4:] if not line.startswith("@@")]

    return lines
