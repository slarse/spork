import subprocess
import itertools
import os
import sys
import time
import pathlib
import dataclasses
import collections
import enum
import contextlib
import tempfile
import shutil
import hashlib
from typing import List, Generator, Iterable, Tuple, Optional, Iterator

import git
import daiquiri

from . import gitutils
from . import fileutils
from . import containers as conts
from . import javautils
from . import procutils

LOGGER = daiquiri.getLogger(__name__)


def run_file_merges(
    file_merge_dirs: List[pathlib.Path], merge_cmd: str
) -> Iterable[conts.MergeResult]:
    """Run the file merges in the provided directories and put the output in a
    file called `merge_cmd`.java.

    Args:
        file_merge_dirs: A list of directories with file merge scenarios. Each
            directory must contain Base.java, Left.java, Right.java and
            Expected.java
        merge_cmd: The merge command to execute. Will be called as
            `merge_cmd Left.java Base.java Right.java -o merge_cmd.java`.
    Returns:
        A generator that yields one conts.MergeResult per merge directory.
    """
    yield from _run_file_merges(file_merge_dirs, merge_cmd)


def run_individual_file_merge(
    merge_dir: pathlib.Path, merge_cmd: str
) -> conts.MergeResult:
    """Run a single file merge using the specified merge command.

    Args:
        merge_dir: A merge directory containing Base.java, Left.java,
            Right.java and Expected.java
        merge_cmd: The merge command to execute on the merge scenario
    Returns:
        A MergeResult.
    """
    sanitized_merge_cmd = pathlib.Path(merge_cmd).name.replace(" ", "_")
    filenames = [f.name for f in merge_dir.iterdir() if f.is_file()]

    def get_filename(prefix: str) -> str:
        matches = [name for name in filenames if name.startswith(prefix)]
        assert len(matches) == 1
        return matches[0]

    merge_file = merge_dir / f"{sanitized_merge_cmd}.java"
    base = merge_dir / get_filename("Base")
    left = merge_dir / get_filename("Left")
    right = merge_dir / get_filename("Right")
    expected = merge_dir / get_filename("Expected")

    assert base.is_file()
    assert left.is_file()
    assert right.is_file()
    assert expected.is_file()

    outcome, runtime = _run_file_merge(
        merge_dir,
        merge_cmd,
        base=base,
        left=left,
        right=right,
        expected=expected,
        merge=merge_file,
    )
    return conts.MergeResult(
        merge_dir=merge_dir,
        merge_file=merge_file,
        base_file=base,
        left_file=left,
        right_file=right,
        expected_file=expected,
        merge_cmd=sanitized_merge_cmd,
        outcome=outcome,
        runtime=runtime,
    )


def _run_file_merges(file_merge_dirs: List[pathlib.Path], merge_cmd: str) -> Iterable:
    for merge_dir in file_merge_dirs:
        yield run_individual_file_merge(merge_dir, merge_cmd)


def _run_file_merge(scenario_dir, merge_cmd, base, left, right, expected, merge):
    timed_out = False
    start = time.perf_counter()

    proc = None
    out = None

    try:
        proc, out = procutils.run_with_sigkill_timeout(
            f"{merge_cmd} {left} {base} {right} -o {merge}".split(),
            timeout=gitutils.MERGE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        LOGGER.exception(merge_cmd)
        timed_out = True
        LOGGER.error(f"{merge_cmd} timed out")
    except:
        LOGGER.exception(f"error running {merge_cmd}")

    runtime = time.perf_counter() - start if not timed_out else gitutils.MERGE_TIMEOUT

    if not merge.is_file():
        LOGGER.error(
            f"{merge_cmd} failed to produce a merge file on {scenario_dir.parent.name}/{scenario_dir.name}"
        )
        if proc != None:
            LOGGER.error(out)
        return conts.MergeOutcome.FAIL, runtime
    elif proc is None or proc.returncode != 0:
        LOGGER.warning(
            f"Merge conflict in {scenario_dir.parent.name}/{scenario_dir.name}"
        )
        return conts.MergeOutcome.CONFLICT, runtime
    else:
        LOGGER.info(
            f"Successfully merged {scenario_dir.parent.name}/{scenario_dir.name}"
        )
        return conts.MergeOutcome.SUCCESS, runtime


def run_git_merges(
    merge_scenarios: List[conts.MergeScenario],
    merge_drivers: List[str],
    repo: git.Repo,
    build: bool,
    base_eval_dir: Optional[pathlib.Path] = None,
) -> Iterable[conts.GitMergeResult]:
    """Replay the provided merge scenarios using git-merge. Assumes that the
    merge scenarios belong to the provided repo. The merge drivers must be
    defined in the global .gitconfig file, https://github.com/kth/spork for
    details on that.

    Args:
        merge_scenarios: A list of merge scenarios.
        merge_drivers: A list of merge driver names to execute the merge with.
            Each driver must be defined in the global .gitconfig file.
        repo: The related repository.
        build: If True, try to build the project with Maven after merge.
        base_eval_dir: If specified, run Java bytecode evaluation in the given
            directory. Implies build.
    Returns:
        An iterable of merge results.
    """
    for ms in merge_scenarios:
        LOGGER.info(
            f"Replaying merge commit {ms.expected.hexsha}, "
            f"base: {ms.base.hexsha} left: {ms.left.hexsha} "
            f"right: {ms.right.hexsha}"
        )
        yield from run_git_merge(ms, merge_drivers, repo, build, base_eval_dir)


def run_git_merge(
    merge_scenario: conts.MergeScenario,
    merge_drivers: List[str],
    repo: git.Repo,
    build: bool,
    base_eval_dir: Optional[pathlib.Path] = None,
) -> Iterator[conts.GitMergeResult]:
    """Replay a single merge scenario. Assumes that the merge scenario belongs
    to the provided repo. The merge tool to use must be configured in
    .gitattributes and .gitconfig, see the README at
    https://github.com/kth/spork for details on that.

    Args:
        merge_scenario: A merge scenario.
        merge_drivers: One or more merge driver names. Each merge driver must
            be defined in the global .gitconfig file.
        repo: The related repository.
        build: If True, try to build the project with Maven after merge.
    Returns:
        An iterable of GitMergeResults, one for each driver
    """
    ms = merge_scenario  # alias for less verbosity
    expected_classfiles = tuple()

    eval_dir = base_eval_dir / merge_scenario.expected.hexsha if base_eval_dir else None

    if eval_dir:
        with gitutils.saved_git_head(repo):
            changed_sourcefiles = [
                file_merge.expected.path
                for file_merge in gitutils.extract_conflicting_files(
                    repo, merge_scenario
                )
            ]
            LOGGER.info(f"Found changed source files: {changed_sourcefiles}")
            expected_classfiles = _extract_expected_revision_classfiles(
                repo, ms, eval_dir, changed_sourcefiles
            )
            if not expected_classfiles:
                LOGGER.warning(
                    "Found no expected classfiles for merge scenario "
                    f"{ms.expected.hexsha}, skipping ..."
                )
                return

    for merge_driver in merge_drivers:
        build_ok = False

        with gitutils.merge_no_commit(
            repo,
            ms.left.hexsha,
            ms.right.hexsha,
            driver_config=(merge_driver, "*.java"),
        ) as merge_stat:
            merge_ok, _ = merge_stat
            _log_cond(
                "Merge replay OK",
                "Merge conflict or failure",
                use_info=merge_ok,
            )

            if build or eval_dir:
                LOGGER.info("Building replayed revision")
                build_ok, output = javautils.mvn_compile(workdir=repo.working_tree_dir)
                if eval_dir:
                    (eval_dir / f"{merge_driver}_build_output.txt").write_bytes(output)
                _log_cond(
                    "Replayed build OK",
                    output,
                    use_info=build_ok,
                )

            if eval_dir:
                for (classfile_pair, equal,) in javautils.compare_compiled_bytecode(
                    pathlib.Path(repo.working_tree_dir),
                    expected_classfiles,
                    eval_dir,
                    merge_driver,
                ):
                    expected_classfile_relpath = (
                        classfile_pair.expected.original_relpath
                    )
                    expected_src_relpath = classfile_pair.expected.original_src_relpath
                    expected_classfile_qualname = classfile_pair.expected.qualname
                    classfile_dir = classfile_pair.expected.copy_basedir.relative_to(
                        base_eval_dir
                    )
                    yield conts.GitMergeResult(
                        merge_commit=ms.expected.hexsha,
                        classfile_dir=classfile_dir,
                        expected_classfile_relpath=expected_classfile_relpath,
                        expected_classfile_qualname=expected_classfile_qualname,
                        expected_src_relpath=expected_src_relpath,
                        merge_driver=merge_driver,
                        build_ok=build_ok,
                        merge_ok=merge_ok,
                        eval_ok=equal,
                    )
            else:
                yield conts.GitMergeResult(
                    merge_commit=ms.expected.hexsha,
                    merge_driver=merge_driver,
                    merge_ok=merge_ok,
                    build_ok=build_ok,
                    classfile_dir="",
                    expected_classfile_relpath="",
                    expected_src_relpath="",
                    expected_classfile_qualname="",
                    eval_ok=False,
                )


def _extract_expected_revision_classfiles(
    repo: git.Repo,
    ms: conts.MergeScenario,
    eval_dir: pathlib.Path,
    unmerged_files: List[pathlib.Path],
) -> List[conts.ExpectedClassfile]:
    """Extract expected classfiles, copy them to the evaluation directory,
    return a list of tuples with the absolute path to the copy and the path to
    the original classfile relative to the repository root.
    """
    gitutils.checkout_clean(repo, ms.expected.hexsha)
    LOGGER.info("Building expected revision")
    worktree_dir = pathlib.Path(repo.working_tree_dir)

    build_ok, build_output = javautils.mvn_compile(workdir=worktree_dir)
    if not build_ok:
        LOGGER.error(build_output.decode(sys.getdefaultencoding()))
        raise RuntimeError(f"Failed to build expected revision {ms.expected.hexsha}")

    sources = [worktree_dir / unmerged for unmerged in unmerged_files]
    LOGGER.info(f"Extracted unmerged files: {sources}")

    expected_classfiles = []
    for src, classfiles, pkg in (
        (src, *javautils.locate_classfiles(src, basedir=worktree_dir))
        for src in sources
    ):
        for classfile in classfiles:
            original_relpath = classfile.relative_to(worktree_dir)
            original_src_relpath = src.relative_to(worktree_dir)
            qualname = f"{pkg}.{classfile.stem}"
            copy_basedir = eval_dir / fileutils.create_unique_filename(
                path=original_relpath, name=classfile.name
            )
            classfile_copy = javautils.copy_to_pkg_dir(
                classfile, pkg, copy_basedir / "expected"
            )
            tup = conts.ExpectedClassfile(
                copy_abspath=classfile_copy,
                copy_basedir=copy_basedir,
                qualname=qualname,
                original_relpath=original_relpath,
                original_src_relpath=original_src_relpath,
            )
            expected_classfiles.append(tup)

    LOGGER.info(f"Extracted classfiles: {expected_classfiles}")

    for classfile in expected_classfiles:
        LOGGER.info(
            f"Removing duplicate checkcasts from expected revision of {classfile.copy_abspath.name}"
        )
        javautils.remove_duplicate_checkcasts(classfile.copy_abspath)

    return expected_classfiles


def _log_cond(info: str, warning: str, use_info: bool):
    if use_info:
        LOGGER.info(info)
    else:
        LOGGER.warning(warning)


def is_buildable(commit_sha: str, repo: git.Repo) -> bool:
    """Try to build the commit with Maven.

    Args:
        commit_sha: A commit hexsha.
        repo: The related Git repo.
    Returns:
        True if the build was successful.
    """
    with gitutils.saved_git_head(repo):
        repo.git.checkout(commit_sha, "--force")
        LOGGER.info(f"Building commit {commit_sha}")
        build_ok, _ = javautils.mvn_compile(workdir=repo.working_tree_dir)
        return build_ok


def is_testable(commit_sha: str, repo: git.Repo) -> bool:
    """Try to run the project's test suite with Maven.

    Args:
        commit_sha: A commit hexsha.
        repo: The related Git repo.
    Returns:
        True if the build was successful.
    """
    with gitutils.saved_git_head(repo):
        repo.git.checkout(commit_sha, "--force")
        return javautils.mvn_test(workdir=repo.working_tree_dir)


def run_running_time_benchmark(
    reference_merge_results: List[conts.NamedMergeEvaluation],
    merge_dirs_root: pathlib.Path,
    num_repetitions: int,
) -> Iterable[conts.MergeResult]:
    _verify_merge_scenarios_exist_in_merge_dir(reference_merge_results, merge_dirs_root)
    return (
        _merge_and_verify_result(merge_dirs_root, reference_eval)
        for reference_eval in reference_merge_results
        for _ in range(0, num_repetitions)
    )


def _verify_merge_scenarios_exist_in_merge_dir(
    reference_merge_results: List[conts.NamedMergeEvaluation],
    merge_dirs_root: pathlib.Path,
):
    LOGGER.info("Validating merge directories against reference results ...")
    for merge_eval in reference_merge_results:
        merge_dir_abspath = _get_merge_dir_abspath(merge_dirs_root, merge_eval)
        expected_filesnames = [
            "Base.java",
            "Left.java",
            "Right.java",
        ]
        assert merge_dir_abspath.is_dir(), f"{merge_dir_abspath} does not exist"
        _assert_matches_hash(merge_dir_abspath / "Base.java", merge_eval.base_blob)
        _assert_matches_hash(merge_dir_abspath / "Left.java", merge_eval.left_blob)
        _assert_matches_hash(merge_dir_abspath / "Right.java", merge_eval.right_blob)

        if merge_eval.outcome == conts.MergeOutcome.SUCCESS:
            _assert_matches_hash(
                merge_dir_abspath / f"{merge_eval.merge_cmd}.java",
                merge_eval.replayed_blob,
            )

    LOGGER.info("SUCCESS: All merge directories accounted for")


def _merge_and_verify_result(
    merge_dirs_root: pathlib.Path, reference_merge_eval: conts.NamedMergeEvaluation
) -> conts.MergeResult:
    reference_merge_dir_abspath = _get_merge_dir_abspath(
        merge_dirs_root, reference_merge_eval
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = pathlib.Path(tmpdir)
        merge_dir = (
            workdir
            / "merge_dirs"
            / reference_merge_dir_abspath.parent.parent.name
            / reference_merge_dir_abspath.parent.name
            / reference_merge_dir_abspath.name
        )
        merge_dir.mkdir(parents=True, exist_ok=False)
        _copy_merge_dir(src=reference_merge_dir_abspath, dst=merge_dir)

        with _in_workdir(workdir):
            merge_result = run_individual_file_merge(
                merge_dir.relative_to(workdir), reference_merge_eval.merge_cmd
            )

            if merge_result.outcome != conts.MergeOutcome.FAIL and (
                gitutils.hash_object(merge_result.merge_file)
                != reference_merge_eval.replayed_blob
            ):
                raise RuntimeError(
                    f"Content mismatch in merge file for {reference_merge_eval}"
                )

    return merge_result


def _assert_matches_hash(filepath: pathlib.Path, expected_hash: str):
    actual_hash = gitutils.hash_object(filepath)
    assert actual_hash == expected_hash, f"hash mismatch for {filepath}"


def _copy_merge_dir(src: pathlib.Path, dst: pathlib.Path):
    for filename in ["Left.java", "Right.java", "Base.java", "Expected.java"]:
        shutil.copyfile(src / filename, dst / filename)


def _get_merge_dir_abspath(
    merge_dirs_root: pathlib.Path, merge_eval: conts.NamedMergeEvaluation
) -> pathlib.Path:
    return merge_dirs_root / merge_eval.project.replace("/", "_") / merge_eval.merge_dir


@contextlib.contextmanager
def _in_workdir(workdir: pathlib.Path):
    orig_dir = os.getcwd()
    try:
        os.chdir(str(workdir))
        yield
    finally:
        os.chdir(orig_dir)
