"""Top-level commands that correspond to the CLI commands."""
import argparse
import pathlib
import sys
import itertools
import dataclasses
import re
import multiprocessing
import tempfile
import shutil
import time
import collections

from typing import List, Optional, Iterable, Mapping, Callable

import daiquiri
import git
import numpy as np
import pandas as pd

from . import evaluate
from . import run
from . import gitutils
from . import fileutils
from . import reporter
from . import mpi
from . import containers as conts


LOGGER = daiquiri.getLogger(__name__)


def run_file_merges(
    repo_name: str,
    github_user: Optional[str],
    eval_func: Callable,
    output_file: Optional[pathlib.Path],
    use_mpi: bool,
    merge_scenarios: Optional[pathlib.Path],
    num_merges: Optional[int],
    gather_metainfo: bool,
    base_merge_dir: pathlib.Path = pathlib.Path("merge_directory"),
):
    """Run individual file merges."""
    expected_merge_scenarios = (
        reporter.read_csv(merge_scenarios, container=conts.SerializableMergeScenario)
        if merge_scenarios
        else None
    )
    evaluations, file_merges, merge_dirs = _run_file_merges(
        eval_func=eval_func,
        repo_name=repo_name,
        github_user=github_user,
        num_merges=num_merges,
        use_mpi=use_mpi,
        expected_merge_scenarios=expected_merge_scenarios,
        base_merge_dir=base_merge_dir,
    )
    output_file = output_file
    reporter.write_csv(
        data=evaluations,
        container=conts.MergeEvaluation,
        dst=output_file,
    )

    if gather_metainfo:
        _output_java_blob_metainfos(merge_dirs, base_output_file=output_file)
        _output_file_merge_metainfos(file_merges, base_output_file=output_file)


def file_merge_metainfo_path(base_output_path: pathlib.Path) -> pathlib.Path:
    """Return the expected path of a merge metainfo file corresponding to the
    base output file."""
    return base_output_path.parent / (
        base_output_path.stem + "_file_merge_metainfo.csv"
    )


def blob_metainfo_path(base_output_path: pathlib.Path) -> pathlib.Path:
    """Return the expected path of a blob metainfo file corresponding to the
    base output file."""
    return base_output_path.parent / (base_output_path.stem + "_blob_metainfo.csv")


def _output_java_blob_metainfos(merge_dirs, base_output_file):
    LOGGER.info("Gathering Java blob metainfo")
    metainfo_output_file = blob_metainfo_path(base_output_file)
    metainfos = evaluate.gather_java_blob_metainfos(merge_dirs)
    reporter.write_csv(
        data=metainfos,
        container=conts.JavaBlobMetainfo,
        dst=metainfo_output_file,
    )
    LOGGER.info(f"Java blob metainfo written to {metainfo_output_file}")


def _output_file_merge_metainfos(file_merges, base_output_file):
    LOGGER.info("Gathering file merge metainfo")
    metainfo_output_file = file_merge_metainfo_path(base_output_file)
    file_merge_metainfos = list(
        map(conts.FileMergeMetainfo.from_file_merge, file_merges)
    )
    reporter.write_csv(
        data=file_merge_metainfos,
        container=conts.FileMergeMetainfo,
        dst=metainfo_output_file,
    )
    LOGGER.info(f"File merge metainfo written to {metainfo_output_file}")


def extract_merge_scenarios(
    repo_name: str,
    github_user: str,
    output_file: pathlib.Path,
    non_trivial: bool,
    buildable: bool,
    testable: bool,
    skip_non_content_conflicts: bool,
    shuffle_commits: bool = False,
    timeout: Optional[int] = None,
):
    """Extract merge commits."""
    original_repo = _get_repo(repo_name, github_user)
    commit_shas = list(gitutils.extract_merge_commit_shas(original_repo))

    if shuffle_commits:
        np.random.shuffle(commit_shas)

    # split the workload
    num_procs = multiprocessing.cpu_count() // 2  # assume SMT
    LOGGER.info(f"Using {num_procs} CPUs")

    pool = multiprocessing.Pool(num_procs)
    results = []

    chunk_size = len(commit_shas) // num_procs
    num_chunks = num_procs
    deadline = time.perf_counter() + timeout if timeout else None
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i != num_procs - 1 else len(commit_shas)

        commits_chunk = commit_shas[start:end]

        kwds = dict(
            original_repo_path=pathlib.Path(original_repo.working_dir),
            commit_shas=commits_chunk,
            non_trivial=non_trivial,
            buildable=buildable,
            testable=testable,
            skip_non_content_conflicts=skip_non_content_conflicts,
            deadline=deadline,
        )
        result = pool.apply_async(_extract_merge_scenarios, kwds=kwds)
        results.append(result)

    pool.close()
    pool.join()

    serializable_merge_scenarios = []
    for result in results:
        try:
            serializable_merge_scenarios += result.get()
        except:
            LOGGER.exception(
                f"Exception when extracting commits for {github_user}/{repo_name}"
            )

    reporter.write_csv(
        data=serializable_merge_scenarios,
        container=conts.SerializableMergeScenario,
        dst=output_file,
    )
    LOGGER.info(f"Merge commits saved to {output_file}")


def _extract_merge_scenarios(
    original_repo_path: pathlib.Path,
    commit_shas: List[str],
    non_trivial: bool,
    buildable: bool,
    testable: bool,
    skip_non_content_conflicts: bool,
    deadline: Optional[bool],
) -> List[conts.SerializableMergeScenario]:
    with tempfile.TemporaryDirectory() as repo_dir:
        repo_path = pathlib.Path(repo_dir) / "repo"
        repo = git.Repo.clone_from(str(original_repo_path), to_path=str(repo_path))

        merge_scenarios = gitutils.extract_merge_scenarios(
            repo, non_trivial=non_trivial, merge_commit_shas=commit_shas
        )

        def is_buildable(ms: conts.MergeScenario) -> bool:
            return all(
                run.is_buildable(commit.hexsha, repo)
                for commit in [ms.base, ms.left, ms.right, ms.expected]
            )

        serializable_merge_scenarios = []
        for ms in merge_scenarios:
            if deadline is not None and time.perf_counter() > deadline:
                LOGGER.warning(
                    f"Timed out collecting merge scenarios from {original_repo_path}"
                )
                break

            if skip_non_content_conflicts and gitutils.contains_non_content_conflict(
                repo, ms
            ):
                continue
            if (buildable or testable) and not is_buildable(ms):
                continue
            if testable and not run.is_testable(ms.expected.hexsha, repo):
                continue

            serializable = conts.SerializableMergeScenario.from_merge_scenario(ms)
            serializable_merge_scenarios.append(serializable)

        return serializable_merge_scenarios


def extract_file_merge_metainfo(
    repo_name: str,
    github_user: Optional[str],
    merge_scenarios: Optional[pathlib.Path],
    output_file: pathlib.Path,
    num_merges: Optional[int],
):
    """Extract metainfo about the file merges."""
    repo = _get_repo(repo_name, github_user)
    serializable_merge_scenarios = (
        reporter.read_csv(
            container=conts.SerializableMergeScenario, csv_file=merge_scenarios
        )
        if merge_scenarios
        else None
    )
    expected_merge_scenarios = _get_merge_scenarios(repo, serializable_merge_scenarios)

    file_merges = gitutils.extract_all_conflicting_files(repo, expected_merge_scenarios)
    file_merge_metainfos = list(
        map(conts.FileMergeMetainfo.from_file_merge, file_merges)
    )[:num_merges]
    reporter.write_csv(
        data=file_merge_metainfos,
        container=conts.FileMergeMetainfo,
        dst=output_file,
    )


def git_merge(
    repo_name: str,
    github_user: Optional[str],
    merge_drivers: List[str],
    merge_scenarios: pathlib.Path,
    output_file: pathlib.Path,
    build: bool,
    base_eval_dir: Optional[pathlib.Path],
    num_merges: Optional[int],
):
    """Run git merge on all scenarios."""
    repo = _get_repo(repo_name, github_user)

    serializable_merge_scenarios = reporter.read_csv(
        container=conts.SerializableMergeScenario, csv_file=merge_scenarios
    )
    expected_merge_scenarios = _get_merge_scenarios(repo, serializable_merge_scenarios)
    merge_results = run.run_git_merges(
        expected_merge_scenarios,
        merge_drivers,
        repo,
        build,
        base_eval_dir,
    )
    reporter.write_csv(
        data=merge_results, container=conts.GitMergeResult, dst=output_file
    )


def measure_running_times(
    reference_merge_results_file: pathlib.Path,
    base_merge_dir: pathlib.Path,
    num_repetitions: int,
):
    reference_merge_results = reporter.read_csv(
        reference_merge_results_file, conts.NamedMergeEvaluation
    )
    merge_cmds = sorted(
        {merge_eval.merge_cmd for merge_eval in reference_merge_results}
    )
    LOGGER.info(
        f"Loaded {len(reference_merge_results)} reference merge results for {merge_cmds}"
    )
    _verify_merge_scenarios_exist_in_merge_dir(reference_merge_results, base_merge_dir)

    LOGGER.info(f"Running benchmark with {num_repetitions} repetitions")
    merge_results = _run_runtime_benchmarks(
        reference_merge_results, base_merge_dir, num_repetitions
    )

    reporter.write_csv(
        data=merge_results, container=conts.MergeResult, dst="runtime_benchmark.csv"
    )


def _verify_merge_scenarios_exist_in_merge_dir(
    reference_merge_results: List[conts.NamedMergeEvaluation],
    base_merge_dir: pathlib.Path,
):
    LOGGER.info("Validating merge directories against reference results ...")
    for merge_eval in reference_merge_results:
        merge_dir_abspath = _get_merge_dir_abspath(base_merge_dir, merge_eval)
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


def _assert_matches_hash(filepath: pathlib.Path, expected_hash: str):
    actual_hash = gitutils.hash_object(filepath)
    assert actual_hash == expected_hash, f"hash mismatch for {filepath}"


def _run_runtime_benchmarks(
    reference_merge_results: List[conts.NamedMergeEvaluation],
    base_merge_dir: pathlib.Path,
    num_repetitions: int,
) -> Iterable[conts.MergeResult]:
    return (
        _merge_and_verify_result(base_merge_dir, reference_eval)
        for reference_eval in reference_merge_results
        for _ in range(0, num_repetitions)
    )


def _copy_merge_scenario(from_dir: pathlib.Path, to_dir: pathlib.Path):
    for filename in ["Left.java", "Right.java", "Base.java", "Expected.java"]:
        shutil.copyfile(from_dir / filename, to_dir / filename)


def _merge_and_verify_result(
    base_merge_dir: pathlib.Path, reference_merge_eval: conts.NamedMergeEvaluation
) -> conts.MergeResult:
    reference_merge_dir_abspath = _get_merge_dir_abspath(
        base_merge_dir, reference_merge_eval
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
        _copy_merge_scenario(reference_merge_dir_abspath, merge_dir)

        with _in_workdir(workdir):
            merge_result = run.run_individual_file_merge(
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


def _get_merge_dir_abspath(
    base_merge_dir: pathlib.Path, merge_eval: conts.NamedMergeEvaluation
) -> pathlib.Path:
    return base_merge_dir / merge_eval.project.replace("/", "_") / merge_eval.merge_dir

import os
import contextlib

@contextlib.contextmanager
def _in_workdir(workdir: pathlib.Path):
    orig_dir = os.getcwd()
    try:
        os.chdir(str(workdir))
        yield
    finally:
        os.chdir(orig_dir)


def runtime_benchmark(
    repo_name: str,
    github_user: Optional[str],
    merge_commands: List[str],
    num_runs: int,
    file_merge_metainfo: pathlib.Path,
    base_merge_dir: pathlib.Path,
    num_merges: Optional[int],
    output_file: pathlib.Path,
):
    """Run a runtime benchmark on individual file merges."""
    repo = _get_repo(repo_name, github_user)
    file_merge_metainfo = reporter.read_csv(
        csv_file=file_merge_metainfo, container=conts.FileMergeMetainfo
    )
    file_merges = (conts.FileMerge.from_metainfo(repo, m) for m in file_merge_metainfo)
    merge_dirs = fileutils.create_merge_dirs(base_merge_dir, file_merges)[:num_merges]

    runtime_results = itertools.chain.from_iterable(
        run.runtime_benchmark(merge_dirs, merge_cmd, num_runs)
        for merge_cmd in merge_commands
    )

    reporter.write_csv(
        data=runtime_results, container=conts.RuntimeResult, dst=output_file
    )


def num_core_contributors(repo_name: str, github_user: Optional[str], threshold: float):
    """Calculate the number of core contributors."""
    repo = _get_repo(repo_name, github_user)
    num_core_contributors = gitutils.num_core_contributors(repo, threshold)
    print(
        f"Amount of core contributors for {github_user}/{repo_name}: {num_core_contributors}"
    )


def _run_file_merges(
    eval_func: Callable,
    repo_name: str,
    github_user: str,
    num_merges: Optional[int],
    use_mpi: bool,
    expected_merge_scenarios: Optional[List[conts.SerializableMergeScenario]],
    base_merge_dir: pathlib.Path = pathlib.Path("merge_directory"),
) -> (Iterable[conts.MergeEvaluation], List[conts.FileMerge]):
    assert not use_mpi or mpi.RANK == mpi.MASTER_RANK

    repo = _get_repo(repo_name, github_user)
    merge_scenarios = _get_merge_scenarios(repo, expected_merge_scenarios)

    LOGGER.info(f"Found {len(merge_scenarios)} merge scenarios")

    base_merge_dir.mkdir(parents=True, exist_ok=True)
    file_merges = list(gitutils.extract_all_conflicting_files(repo, merge_scenarios))[
        :num_merges
    ]
    merge_dirs = fileutils.create_merge_dirs(base_merge_dir, file_merges)

    LOGGER.info(f"Extracted {len(merge_dirs)} file merges")

    if use_mpi:
        evaluations = mpi.master(merge_dirs)
    else:
        evaluations = eval_func(merge_dirs)

    return evaluations, file_merges, merge_dirs


def _get_repo(repo: str, github_user: Optional[str]) -> git.Repo:
    if github_user is not None:
        return gitutils.clone_repo(repo, github_user)
    else:
        return git.Repo(repo)


def _get_merge_scenarios(
    repo: git.Repo,
    serializable_scenarios: Optional[List[conts.SerializableMergeScenario]],
) -> List[conts.MergeScenario]:
    return (
        [
            conts.MergeScenario.from_serializable(repo, serializable)
            for serializable in serializable_scenarios
        ]
        if serializable_scenarios is not None
        else gitutils.extract_merge_scenarios(repo)
    )
