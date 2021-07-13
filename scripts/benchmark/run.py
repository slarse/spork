import multiprocessing
import functools
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
import pandas as pd

from . import gitutils
from . import fileutils
from . import containers as conts
from . import javautils
from . import procutils
from . import evaluate

LOGGER = daiquiri.getLogger(__name__)


def run_file_merges(
    file_merge_dirs: List[pathlib.Path], merge_commands: str
) -> Iterable[conts.MergeResult]:
    """Run the file merges in the provided directories and put the output in a
    file called `merge_cmd`.java.

    Args:
        file_merge_dirs: A list of directories with file merge scenarios. Each
            directory must contain Base.java, Left.java, Right.java and
            Expected.java
        merge_commands: The merge commands to execute. Will be called as
            `merge_cmd Left.java Base.java Right.java -o merge_cmd.java`.
    Returns:
        A generator that yields one conts.MergeResult per merge directory.
    """
    yield from (
        run_individual_file_merge(merge_dir, merge_cmd)
        for merge_dir, merge_cmd in itertools.product(file_merge_dirs, merge_commands)
    )


def run_individual_file_merge(
    merge_dir: pathlib.Path, merge_cmd: str
) -> conts.InternalMergeResult:
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
    return conts.InternalMergeResult(
        merge_commit=fileutils.extract_commit_sha(merge_dir),
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


def _run_file_merges(
    file_merge_dirs: List[pathlib.Path], merge_cmd: str
) -> Iterable[conts.InternalMergeResult]:
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
        LOGGER.error(
            f"{merge_cmd} timed out on {scenario_dir.parent.name}/{scenario_dir.name}"
        )
    except:
        LOGGER.exception(f"error running {merge_cmd}")

    runtime = time.perf_counter() - start

    if timed_out:
        return conts.MergeOutcome.TIMEOUT, runtime
    elif not merge.is_file():
        LOGGER.error(
            f"{merge_cmd} failed to produce a merge file on {scenario_dir.parent.name}/{scenario_dir.name}"
        )
        if proc != None:
            LOGGER.error(out)
        return conts.MergeOutcome.FAIL, runtime
    elif (
        proc is None
        or proc.returncode != 0
        or gitutils.START_CONFLICT in merge.read_text()
    ):
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
    reference_merge_results: List[conts.MergeResult],
    base_merge_dir: pathlib.Path,
    num_repetitions: int,
) -> Iterable[conts.RunningTime]:
    _verify_merge_scenarios_exist_in_merge_dir(reference_merge_results, base_merge_dir)
    success_reference_merge_results = _filter_out_commits_with_fails(
        reference_merge_results
    )
    return (
        _replay_merge(base_merge_dir, reference_eval)
        for reference_eval in success_reference_merge_results
        for _ in range(0, num_repetitions)
    )


def _filter_out_commits_with_fails(
    reference_merge_results: List[conts.MergeResult],
) -> List[conts.MergeResult]:
    LOGGER.info("Filtering out commits where any tool has failure")
    commits_with_fails = {
        fileutils.extract_commit_sha(merge_result.merge_dir)
        for merge_result in reference_merge_results
        if merge_result.outcome == conts.MergeOutcome.FAIL
    }
    LOGGER.info(f"Found {len(commits_with_fails)} commits with failures, removing ...")
    return [
        merge_result
        for merge_result in reference_merge_results
        if fileutils.extract_commit_sha(merge_result.merge_dir)
        not in commits_with_fails
    ]


def _verify_merge_scenarios_exist_in_merge_dir(
    reference_merge_results: List[conts.MergeResult],
    base_merge_dir: pathlib.Path,
):
    LOGGER.info("Validating merge directories against reference results ...")
    for merge_result in reference_merge_results:
        merge_dir_abspath = _get_merge_dir_abspath(base_merge_dir, merge_result)
        assert (
            base_merge_dir / merge_result.merge_dir
        ).is_dir(), f"{merge_dir_abspath} is not a directory"

        expected_files = [
            merge_result.base_file,
            merge_result.left_file,
            merge_result.right_file,
        ] + (
            [merge_result.merge_file]
            if merge_result.outcome != conts.MergeOutcome.FAIL
            else []
        )

        for relpath in expected_files:
            assert (base_merge_dir / relpath).is_file()

    LOGGER.info("SUCCESS: All merge directories accounted for")


def _replay_merge(
    base_merge_dir: pathlib.Path, reference_merge_result: conts.MergeResult
) -> conts.RunningTime:
    reference_merge_dir_abspath = _get_merge_dir_abspath(
        base_merge_dir, reference_merge_result
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = pathlib.Path(tmpdir)
        merge_dir = workdir / reference_merge_result.merge_dir
        merge_dir.mkdir(parents=True, exist_ok=False)
        _copy_merge_dir(src=reference_merge_dir_abspath, dst=merge_dir)

        with _in_workdir(workdir):
            merge_result = run_individual_file_merge(
                merge_dir.relative_to(workdir), reference_merge_result.merge_cmd
            )

    return conts.RunningTime(
        owner=reference_merge_result.owner,
        repo=reference_merge_result.repo,
        commit_sha=fileutils.extract_commit_sha(merge_dir),
        merge_dir=merge_result.merge_dir,
        merge_cmd=merge_result.merge_cmd,
        running_time=merge_result.runtime,
    )


def _assert_matches_hash(filepath: pathlib.Path, expected_hash: str):
    actual_hash = gitutils.hash_object(filepath)
    assert actual_hash == expected_hash, f"hash mismatch for {filepath}"


def _copy_merge_dir(src: pathlib.Path, dst: pathlib.Path):
    for filename in ["Left.java", "Right.java", "Base.java", "Expected.java"]:
        shutil.copyfile(src / filename, dst / filename)


def _get_merge_dir_abspath(
    base_merge_dir: pathlib.Path, merge_result: conts.MergeResult
) -> pathlib.Path:
    return base_merge_dir / merge_result.merge_dir


def run_evaluations(
    reference_merge_results: List[conts.MergeResult],
    base_merge_dir: pathlib.Path,
) -> pd.DataFrame:
    """Run evaluations on performed file merges."""
    _verify_merge_scenarios_exist_in_merge_dir(reference_merge_results, base_merge_dir)
    return _evaluate(reference_merge_results, base_merge_dir)


def _evaluate(
    reference_merge_results: List[conts.MergeResult], base_merge_dir: pathlib.Path
) -> pd.DataFrame:
    num_procs = max(1, multiprocessing.cpu_count() // 2)  # assume SMT
    LOGGER.info(f"Using {num_procs} CPUs")
    pool = multiprocessing.Pool(num_procs)

    evaluate_file_merge_with_basedir = functools.partial(
        evaluate.evaluate_file_merge, base_merge_dir=base_merge_dir
    )
    evaluations = pool.map(evaluate_file_merge_with_basedir, reference_merge_results)
    evaluation_dicts = map(dataclasses.asdict, evaluations)
    return pd.DataFrame(evaluation_dicts)


@contextlib.contextmanager
def _in_workdir(workdir: pathlib.Path):
    orig_dir = os.getcwd()
    try:
        os.chdir(str(workdir))
        yield
    finally:
        os.chdir(orig_dir)
