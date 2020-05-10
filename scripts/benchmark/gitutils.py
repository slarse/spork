"""Module with some Git utility functions and classes."""
import sys
import enum
import dataclasses
import os
import itertools
import pathlib
import shutil
import contextlib
import subprocess

from typing import (
    List,
    Optional,
    Mapping,
    Sequence,
    Tuple,
    Iterable,
    ContextManager,
)

import git
import daiquiri

from . import containers as conts

START_CONFLICT = "<<<<<<<"
MID_CONFLICT = "======="
END_CONFLICT = ">>>>>>>"

LOGGER = daiquiri.getLogger(__name__)


def extract_merge_scenarios(
    repo: git.Repo,
    non_trivial: bool = False,
    merge_commit_shas: Optional[List[str]] = None,
) -> List[conts.MergeScenario]:
    """Extract merge scenarios from a repo.

    Args:
        repo: A Git repo.
        merge_commit_shas: Commit shas to extract scenarios for.
        non_trivial: If true, extract only scenarios with non-disjoint edits to files.
    Returns:
        A list of merge scenarios.
    """
    merge_commits = (
        commit for commit in repo.iter_commits() if len(commit.parents) == 2
    )
    if merge_commit_shas is not None:
        expected_merge_commits = set(merge_commit_shas)
        merge_commits = (
            commit
            for commit in merge_commits
            if commit.hexsha in expected_merge_commits
        )
    else:
        expected_merge_commits = set()

    merge_commits = list(merge_commits)
    merge_scenarios = []

    for merge in merge_commits:
        left, right = merge.parents
        base = repo.merge_base(*merge.parents)

        if not base:
            LOGGER.warning(
                f"No merge base for commits {left.hexsha} and {right.hexsha}"
            )
            continue
        elif len(base) > 1:
            LOGGER.warning(
                f"Ambiguous merge base for commits {left.hexsha} and {right.hexsha}: {base}"
            )
            continue

        scenario = conts.MergeScenario(merge, base[0], left, right)

        if non_trivial and not extract_conflicting_files(repo, scenario):
            LOGGER.warning(f"Skipping trivial merge commit {merge.hexsha}")
        else:
            expected_merge_commits -= {merge.hexsha}
            merge_scenarios.append(scenario)

    if expected_merge_commits:
        msg = f"Missing merge commits: {expected_merge_commits}"
        raise RuntimeError(msg)

    return merge_scenarios


def extract_all_conflicting_files(
    repo: git.Repo, merge_scenarios: Sequence[conts.MergeScenario],
) -> Iterable[conts.FileMerge]:
    return itertools.chain.from_iterable(
        extract_conflicting_files(repo, ms) for ms in merge_scenarios
    )


def extract_conflicting_files(
    repo: git.Repo,
    merge_scenario: conts.MergeScenario,
    skip_conflict_markers: bool = True,
) -> List[conts.FileMerge]:
    LOGGER.info(
        f"Extracting conflicting files for merge {merge_scenario.expected.hexsha}"
    )

    left = merge_scenario.left
    right = merge_scenario.right
    base = merge_scenario.base
    expected = merge_scenario.expected
    merge_idx: git.IndexFile = repo.index.from_tree(repo, base, left, right)

    left_expected_diff = {
        diff.a_blob.hexsha: diff.b_blob
        for diff in left.diff(expected)
        if diff.a_blob
    }
    right_expected_diff = {
        diff.a_blob.hexsha: diff.b_blob
        for diff in right.diff(expected)
        if diff.a_blob
    }
    base_expected_diff = {
        diff.a_blob.hexsha: diff.b_blob
        for diff in base.diff(expected)
        if diff.a_blob
    }

    file_merges = []

    for _, blobs in merge_idx.unmerged_blobs().items():
        rev_map = {}
        for stage, blob in blobs:
            if stage == 1:
                insert(blob, conts.Revision.BASE, base_expected_diff, rev_map)
            elif stage == 2:
                insert(blob, conts.Revision.LEFT, left_expected_diff, rev_map)
            elif stage == 3:
                insert(
                    blob, conts.Revision.RIGHT, right_expected_diff, rev_map
                )
            else:
                raise ValueError("unknown stage " + stage)

        if rev_map.get(conts.Revision.ACTUAL_MERGE) is None:
            LOGGER.warning(
                "Could not find expected revision, skipping: " + str(rev_map)
            )
            continue
        if (
            conts.Revision.LEFT not in rev_map
            or conts.Revision.RIGHT not in rev_map
        ):
            # this is a delete/modify conflict which can't be resolved by
            # file-based merge tool, and so is not useful in the Spork analysis
            LOGGER.warning(
                "Skipping delete/edit file conflict: " + str(rev_map)
            )
            continue

        file_merge = _to_file_merge(rev_map, merge_scenario)

        if not str(file_merge.expected.name).endswith(".java"):
            LOGGER.warning(
                f"{file_merge.expected.name} is not a Java file, skipping"
            )
            continue
        if skip_conflict_markers and _has_conflict_marker(file_merge):
            LOGGER.warning(
                f"Found conflict markers in scenario {expected.hexsha}/{file_merge.expected.hexsha}, skipping"
            )
            continue

        file_merges.append(file_merge)

    if not file_merges:
        LOGGER.info(
            f"No file merges required for merge commit {expected.hexsha}"
        )

    return file_merges


def contains_delete_modify(repo: git.Repo, ms: conts.MergeScenario) -> bool:
    """Check if the merge scenario contains a delete/modify conflict.

    Args:
        repo: A Git repo.
        ms: A merge scenario.
    Returns:
        True iff the scenario contains a delete/modify conflict.
    """
    index = repo.index.from_tree(
        repo, ms.base.hexsha, ms.left.hexsha, ms.right.hexsha
    )
    for filepath, blobs in index.unmerged_blobs().items():
        blob_states = {state for state, _ in blobs}
        # 2 is the "current" (left) branch and 3 is the "other" (right) branch
        if 2 not in blob_states or 3 not in blob_states:
            return True
    return False


def extract_unmerged_files(
    repo: git.Repo, ms: conts.MergeScenario, file_ext: Optional[str] = None
) -> List[pathlib.Path]:
    """Extract a list of paths to all files that can't be trivially merged for
    the given merge scenario.

    Args:
        repo: A git repository.
        ms: A merge scenario.
        file_ext: Limit the search to the given file extension.
    Returns:
        A list of paths to unmerged files, relative to the root of the repository worktree.
    """
    index = repo.index.from_tree(
        repo, ms.base.hexsha, ms.left.hexsha, ms.right.hexsha
    )
    return [
        path
        for k in index.unmerged_blobs().keys()
        if file_ext is None or (path := pathlib.Path(k)).suffix == file_ext
    ]


def _has_conflict_marker(file_merge: conts.FileMerge) -> bool:
    return any(
        map(
            _contains_conflict_marker,
            [
                file_merge.expected,
                file_merge.base,
                file_merge.left,
                file_merge.right,
            ],
        )
    )


def _contains_conflict_marker(blob: git.Blob) -> bool:
    if blob == None:
        return False

    lines = blob.data_stream[-1].read().decode(encoding="utf8").split("\n")
    return any(
        [
            line
            for line in lines
            if line.startswith(START_CONFLICT) or line.startswith(END_CONFLICT)
        ]
    )


@contextlib.contextmanager
def merge_no_commit(
    repo: git.Repo,
    left_sha: str,
    right_sha: str,
    driver_config: Optional[Tuple[str, str]],
) -> ContextManager[bool]:
    """Returns a context manager that on enter performs the desired merge
    without committing, and on exit aborts the merge and restores the repo HEAD to
    the initial state.

    Args:
        repo: A git repo to merge in.
        left_sha: The hexsha of the left commit (the "current" version).
        right_sha: The hexsha of the right commit (the "other" version).
        driver_config: An optional tuple with (merge_driver_name, file_pattern)
            used to set the merge driver for the merge.
    Returns:
        A context manager that yields True on a merge without conflicts.
    """
    try:
        with saved_git_head(repo):
            checkout_clean(repo, left_sha)
            if driver_config:
                LOGGER.info(f"Using merge driver config {driver_config}")
                set_merge_driver(repo, *driver_config)
            try:
                LOGGER.info(f"Merging: left={left_sha} right={right_sha}")
                output = repo.git.merge(right_sha, "--no-commit")
                success = True
            except git.GitCommandError as exc:
                output = str(exc)
                success = False

            yield success, extract_automerged_files(
                output, pathlib.Path(repo.working_tree_dir)
            )
    finally:
        repo.git.reset("--merge")


def extract_automerged_files(
    git_merge_output: str, worktree_root: pathlib.Path, ext=".java"
) -> List[pathlib.Path]:
    """Extract a list of automerged files from the output of a git merge. Must
    run `git merge` with at least info level 2 (which is the default).
    """
    auto_merged = []
    for line in git_merge_output.strip().split("\n"):
        if line.startswith("Auto-merging") and (not ext or line.endswith(ext)):
            auto_merged.append(worktree_root / line[len("Auto-merging ") :])
    return auto_merged


@contextlib.contextmanager
def saved_git_head(repo: git.Repo) -> ContextManager[None]:
    """Create a context manager that automatically restores the repos HEAD when exited.

    Args:
        repo: A Git repository.
    Returns:
        A context manager that restores the Git HEAD on exit.
    """
    saved_head = repo.head.commit.hexsha
    LOGGER.info(f"Repo HEAD saved at {saved_head}")

    exc = None
    try:
        yield
    except BaseException as e:
        exc = e
    finally:
        checkout_clean(repo, saved_head)
        LOGGER.info(f"Restored repo HEAD to {saved_head}")
        if exc is not None:
            raise exc


def checkout_clean(repo: git.Repo, commitish: str) -> None:
    """Checkout to a commit and clean any untracked files and directories."""
    repo.git.checkout(commitish, "--force")
    repo.git.clean("-xfd")


def _to_file_merge(
    rev_map: Mapping[conts.Revision, git.Blob], ms: conts.MergeScenario
) -> conts.FileMerge:
    base = (
        rev_map[conts.Revision.BASE]
        if conts.Revision.BASE in rev_map
        else None
    )
    left = rev_map[conts.Revision.LEFT]
    right = rev_map[conts.Revision.RIGHT]
    expected = rev_map[conts.Revision.ACTUAL_MERGE]
    return conts.FileMerge(
        base=base,
        left=left,
        right=right,
        expected=expected,
        from_merge_scenario=ms,
    )


def insert(blob, rev, diff_map, rev_map):
    rev_map[rev] = blob
    if blob.hexsha in diff_map:
        expected_blob = diff_map[blob.hexsha]
        assert (
            conts.Revision.ACTUAL_MERGE not in rev_map
            or rev_map[conts.Revision.ACTUAL_MERGE] == expected_blob
        )
        if expected_blob != None:
            rev_map[conts.Revision.ACTUAL_MERGE] = expected_blob


def clone_repo(
    repo_name: str, github_user: str, output_dir: Optional[pathlib.Path] = None
) -> git.Repo:
    """Clone a repo from GitHub and put it in
    'output_dir/github_user/repo_name', or just 'github_user/repo_name' if the
    output_dir is not specified.

    If the repo already exists locally, it is returned as-is, or copied to
    output_dir if specified.

    Args:
        repo_name: Name of the repository.
        github_user: Owner of the repository.
        output_dir: A directory to put the cloned repository in.
    Returns:
        A Git repo.
    """
    qualname = f"{github_user}/{repo_name}"

    output_dir = (
        output_dir if output_dir is not None else pathlib.Path(os.getcwd())
    )
    repo_path = output_dir / github_user / repo_name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if not repo_path.exists():
        url = f"https://github.com/{qualname}.git"
        LOGGER.info(f"Cloning repository from {url}")
        repo = git.Repo.clone_from(url, str(repo_path))
        LOGGER.info(f"Repository cloned to {repo_path}")
    else:
        LOGGER.info(f"Using existing local repository at {repo_path}")
        repo = git.Repo(str(repo_path))

    return repo


def hash_object(path: pathlib.Path) -> str:
    """Compute the SHA1 hash of a blob using Git's hash-object command.

    Args:
        path: Path to a file.
    Returns:
        The SHA1 hash of the content of the file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    proc = subprocess.run(
        ["git", "hash-object", str(path)], capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hash-object exited non-zero on {path}")

    return proc.stdout.decode().strip()


def set_merge_driver(
    repo: git.Repo, driver_name: str, file_pattern: str
) -> None:
    """Set the merge driver for the given pattern by overwriting the repo-local .gitattributes file.

    Args:
        repo: A git repo.
        driver_name: Name of the merge driver to use. Must be configured in the
            global .gitconfig file.
        file_pattern: A filename pattern to associate the driver with.
    """
    (pathlib.Path(repo.working_tree_dir) / ".gitattributes").write_text(
        f"{file_pattern} merge={driver_name}",
        encoding=sys.getdefaultencoding(),
    )


def _get_blob(repo: git.Repo, commit_sha: str, blob_sha: str) -> git.Blob:
    commit = repo.commit(commit_sha)
    tree = commit.tree
    return tree[blob_sha]
