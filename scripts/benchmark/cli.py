"""The CLI for the benchmark suite."""
import sys
import os
import pathlib
import git
import argparse
import functools
import collections
import itertools
import dataclasses

from typing import List, Optional, Iterable

import daiquiri
import logging

from . import evaluate
from . import run
from . import gitutils
from . import fileutils
from . import reporter
from . import command
from . import containers as conts

_CI = os.getenv("TRAVIS_BUILD_DIR")


def setup_logging():
    daiquiri.setup(
        level=logging.INFO if not _CI else logging.DEBUG,
        outputs=(
            daiquiri.output.Stream(
                sys.stdout,
                formatter=daiquiri.formatter.ColorFormatter(
                    fmt="%(color)s[%(levelname)s] %(message)s%(color_stop)s"
                ),
            ),
            # daiquiri.output.File(
            #    filename=str("spork_benchmark.log"),
            #    formatter=daiquiri.formatter.ColorFormatter(
            #        fmt="%(asctime)s [PID %(process)d] [%(levelname)s] "
            #        "%(name)s -> %(message)s"
            #    ),
            # ),
        ),
    )


setup_logging()
LOGGER = daiquiri.getLogger(__name__)


def create_cli_parser():
    parser = argparse.ArgumentParser(
        "Spork merge tester",
        description="A little program to help develop Spork!",
    )

    base_parser = argparse.ArgumentParser(add_help=False)

    base_parser.add_argument(
        "-r",
        "--repo",
        help="Name of the repo to run tests on. If -g is not specified, repo "
        "is assumed to be local.",
        type=str,
        required=True,
    )
    base_parser.add_argument(
        "-g",
        "--github-user",
        help="GitHub username to fetch repo from. Is combined with `--repo`"
        "to form a qualified repo name on the form `repo/user`. If this is "
        "not provided, the repo argument is assumend to be a local directory.",
        type=str,
    )

    base_output_parser = argparse.ArgumentParser(add_help=False, parents=[base_parser])
    base_output_parser.add_argument(
        "-n",
        "--num-merges",
        help="Maximum amount of file merges to recreate.",
        type=int,
        default=None,
    )
    base_output_parser.add_argument(
        "-o",
        "--output",
        help="Where to store the output.",
        type=pathlib.Path,
        default=None,
    )

    base_merge_parser = argparse.ArgumentParser(
        add_help=False, parents=[base_output_parser]
    )
    base_merge_parser.add_argument(
        "--base-merge-dir",
        help="Base directory to perform the merges in.",
        type=pathlib.Path,
        default=pathlib.Path("merge_directory"),
    )
    base_merge_parser.add_argument(
        "--merge-commands",
        help="Merge commands to run.",
        type=str,
        required=True,
        nargs="+",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    file_merge_command = subparsers.add_parser(
        "run-file-merges",
        help="Test a merge tool by merging one file at a time.",
        parents=[base_merge_parser],
    )
    file_merge_command.add_argument(
        "--merge-scenarios",
        help="Path to a CSV file with merge scenarios to operate on.",
        default=None,
        type=pathlib.Path,
    )
    file_merge_command.add_argument(
        "--gather-metainfo",
        help="Gather blob and file merge metainfo for all merge scenarios. Outputs to "
        "<OUTPUT_FILE_STEM>_blob_metainfo.csv and "
        "<OUTPUT_FILE_STEM>_file_merge_metainfo.csv.",
        action="store_true",
    )

    git_merge_command = subparsers.add_parser(
        "run-git-merges",
        help="Replay the merge commits provided using Git and the currently "
        "configured merge driver.",
        parents=[base_output_parser],
    )
    git_merge_command.add_argument(
        "--merge-drivers",
        help="Names of merge drivers. Each driver must be configured in the "
        "global .gitconfig file.",
        required=True,
        nargs="+",
        type=str,
    )
    git_merge_command.add_argument(
        "--merge-scenarios",
        help="Path to a CSV file with merge scenarios to operate on.",
        required=True,
        type=pathlib.Path,
    )
    git_merge_command.add_argument(
        "--build",
        help="Try to build the project with Maven after the merge.",
        action="store_true",
    )
    git_merge_command.add_argument(
        "--eval-dir",
        help="If specified, run the Java bytecode evaluation in the given "
        "directory. Implies --build.",
        type=pathlib.Path,
        default=None,
    )

    merge_extractor_command = subparsers.add_parser(
        "extract-merge-scenarios",
        help="Extract merge scenarios from a repo.",
        parents=[base_output_parser],
    )
    merge_extractor_command.add_argument(
        "--non-trivial",
        help="Extract only non-trivial merges. Implies " "--skip-non-content-conflicts",
        action="store_true",
    )
    merge_extractor_command.add_argument(
        "--buildable",
        help="Only extract merge scenarios if every commit involved, "
        "including the merge commit, can be built with maven",
        action="store_true",
    )
    merge_extractor_command.add_argument(
        "--testable",
        help="Only extract merge scenarios all involved commits are buildable, "
        "and the merge commit passes the test suit. Implies --buildable.",
        action="store_true",
    )
    merge_extractor_command.add_argument(
        "--skip-non-content-conflicts",
        help="Skip any merge scenario that contains at least one "
        "non-content conflict, such as rename/rename or delete/modify.",
        action="store_true",
    )
    merge_extractor_command.add_argument(
        "--merge-scenarios",
        help="Path to a file with previously extracted merge scenarios. "
        "If provided, the command selects scenarios from among these "
        "according to the criteria (e.g. --buildable and --testable). "
        "This is useful for adding requirements to previously extracted "
        "scenarios, or simply to re-validate old merge scenarios.",
        type=pathlib.Path,
    )

    file_merge_metainfo_command = subparsers.add_parser(
        "extract-file-merge-metainfo",
        help="Extract metainfo for non-trivial file merges.",
        parents=[base_output_parser],
    )
    file_merge_metainfo_command.add_argument(
        "--merge-scenarios",
        help="Path to a CSV file with merge scenarios to operate on.",
        default=None,
        type=pathlib.Path,
    )

    core_contributors_command = subparsers.add_parser(
        "num-core-contributors",
        help="Calculate the amount of core contributors.",
        description="The core contributors is the smallest set of "
        "contributors responsible for at least a certain fraction "
        "of all commits. The fraction can be set with the "
        "``--threshold`` argument.",
        parents=[base_parser],
    )
    core_contributors_command.add_argument(
        "-t",
        "--threshold",
        help="The threshold for core contributors. Should be a value in the range [0, 1].",
        type=float,
        required=True,
    )

    base_post_file_merge_parser = argparse.ArgumentParser(add_help=False)
    base_post_file_merge_parser.add_argument(
        "-r",
        "--reference-merge-results",
        type=pathlib.Path,
        required=True,
    )
    base_post_file_merge_parser.add_argument(
        "-o",
        "--output",
        help="Where to store the output.",
        type=pathlib.Path,
        default=None,
    )
    base_post_file_merge_parser.add_argument(
        "--base-merge-dir",
        help="Base directory to perform the merges in.",
        type=_abspath,
        default=pathlib.Path("merge_directory"),
    )

    running_times_command = subparsers.add_parser(
        "measure-running-times",
        help="Measure running times for the tools used to produce the provided "
        "merge results.",
        description="Measure running times for merges already computed with "
        "the run-file-merges command. The merge results from each tool in this "
        "experiment are validated against the previously computed results.",
        parents=[base_post_file_merge_parser],
    )
    running_times_command.add_argument("--num-repetitions", type=int, required=True)

    subparsers.add_parser(
        "evaluate-file-merges",
        help="Run evaluations on previously computed file merges.",
        parents=[base_post_file_merge_parser],
    )

    compose_csv_files_command = subparsers.add_parser(
        "compose-csv-files",
        help="Compose CSV results files from individual project benchmarks "
        "into a single CSV file with an added 'project' column to indicate "
        "which project each row is from.",
    )
    compose_csv_files_command.add_argument(
        "--csv-name",
        help="Name of the CSV file to collect from each project (e.g. `file_merge_results.csv`",
        type=str,
        required=True,
    )
    compose_csv_files_command.add_argument(
        "-o",
        "--output",
        help="Where to store the output.",
        type=pathlib.Path,
        default=None,
    )
    compose_csv_files_command.add_argument(
        "--base-merge-dir",
        help="Base directory to perform the merges in.",
        type=_abspath,
        default=pathlib.Path("merge_directory"),
    )

    return parser


def main():
    parser = create_cli_parser()
    args = parser.parse_args(sys.argv[1:])

    if args.command == "extract-merge-scenarios":
        return command.extract_merge_scenarios(
            repo_name=args.repo,
            github_user=args.github_user,
            output_file=args.output or pathlib.Path("merge_scenarios.csv"),
            non_trivial=args.non_trivial,
            buildable=args.buildable,
            testable=args.testable,
            skip_non_content_conflicts=args.skip_non_content_conflicts,
            merge_scenarios=args.merge_scenarios,
        )
    elif args.command == "extract-file-merge-metainfo":
        command.extract_file_merge_metainfo(
            repo_name=args.repo,
            github_user=args.github_user,
            output_file=args.output or pathlib.Path("file_merge_metainfo.csv"),
            num_merges=args.num_merges,
            merge_scenarios=args.merge_scenarios,
        )
        return
    elif args.command == "run-git-merges":
        return command.git_merge(
            repo_name=args.repo,
            github_user=args.github_user,
            merge_drivers=args.merge_drivers,
            merge_scenarios=args.merge_scenarios,
            output_file=args.output or pathlib.Path("merge_results.csv"),
            build=args.build,
            base_eval_dir=args.eval_dir,
            num_merges=args.num_merges,
        )
    elif args.command == "num-core-contributors":
        command.num_core_contributors(
            repo_name=args.repo,
            github_user=args.github_user,
            threshold=args.threshold,
        )
        return
    elif args.command == "measure-running-times":
        command.measure_running_times(
            reference_merge_results_file=args.reference_merge_results,
            base_merge_dir=args.base_merge_dir,
            num_repetitions=args.num_repetitions,
            output_file=args.output or "running_times.csv",
        )
        return
    elif args.command == "compose-csv-files":
        command.compose_csv_files(
            base_merge_dir=args.base_merge_dir,
            results_csv_name=args.csv_name,
            output_file=args.output,
        )
        return
    elif args.command == "evaluate-file-merges":
        command.evaluate_file_merges(
            args.reference_merge_results,
            args.base_merge_dir,
            args.output or "file_merge_evaluations.csv",
        )
    elif args.command == "run-file-merges":
        command.run_file_merges(
            repo_name=args.repo,
            github_user=args.github_user,
            merge_scenarios=args.merge_scenarios,
            merge_commands=args.merge_commands,
            num_merges=args.num_merges,
            gather_metainfo=args.gather_metainfo,
            output_file=args.output or pathlib.Path("file_merges.csv"),
            base_merge_dir=args.base_merge_dir,
        )
        return
    else:
        raise ValueError(f"Unexpected command: {args.command}")

def _abspath(filepath: str) -> pathlib.Path:
    return pathlib.Path(filepath).resolve(strict=False)

if __name__ == "__main__":
    main()
