"""Experiment module to evaluate Jav amerge tools."""
import pathlib
import sys
import functools
import random
import logging
import shutil

import numpy as np

import daiquiri

from benchmark import command
from benchmark import evaluate

NUM_SCENARIOS = 1000
MAX_SCENARIOS_PER_PROJECT = 50
SCENARIO_EXTRACTION_TIME_LIMIT = 60 * 60  # 1 hour
NUM_RUNNING_TIME_REPS = 10
MERGE_COMMANDS = ("spork", "jdime", "jdimeimproved")
MERGE_DRIVERS = ("spork", "jdime", "jdimeimproved")
CANDIDATE_PROJECTS_FILE = pathlib.Path("buildable_candidates.txt")
BASE_EXPERIMENT_DIRECTORY = pathlib.Path("merge_dirs")

FILE_MERGE_RESULTS_FILENAME = "file_merge_results.csv"
GIT_MERGE_RESULTS_FILENAME = "git_merge_results.csv"
RUNNING_TIMES_FILENAME = "running_times.csv"
EVALUATION_FILENAME = "file_merge_evaluations.csv"

# add projects here if a restart was forced and these need to be reprocessed,
# or if some projects should simply always be processed
# Example: MANDATORY_PROJECTS = ["awslabs/aws-codedeploy-plugin", "inria/spoon"]
MANDATORY_PROJECTS = []


def main():
    """Run the experiments."""
    setup_logging()

    candidates = [
        stripped.split("/")
        for line in CANDIDATE_PROJECTS_FILE.read_text().strip().split("\n")
        if (stripped := line.strip()) not in MANDATORY_PROJECTS
    ]
    np.random.shuffle(candidates)

    projects = [
        mandatory.strip().split("/") for mandatory in MANDATORY_PROJECTS
    ] + candidates

    total_num_merge_scenarios = 0

    for (github_user, repo_name) in projects:
        project_merge_scenarios = run_file_merges_on_project(
            repo_name=repo_name, github_user=github_user
        )
        total_num_merge_scenarios += project_merge_scenarios
        if total_num_merge_scenarios >= NUM_SCENARIOS:
            break

    composed_file_merge_results_file = (
        BASE_EXPERIMENT_DIRECTORY / FILE_MERGE_RESULTS_FILENAME
    )
    command.compose_csv_files(
        base_merge_dir=BASE_EXPERIMENT_DIRECTORY,
        results_csv_name=FILE_MERGE_RESULTS_FILENAME,
        output_file=composed_file_merge_results_file,
    )

    command.evaluate_file_merges(
        reference_merge_results_file=composed_file_merge_results_file,
        base_merge_dir=BASE_EXPERIMENT_DIRECTORY,
        output_file=BASE_EXPERIMENT_DIRECTORY / EVALUATION_FILENAME,
    )

    command.measure_running_times(
        reference_merge_results_file=composed_file_merge_results_file,
        base_merge_dir=BASE_EXPERIMENT_DIRECTORY,
        num_repetitions=NUM_RUNNING_TIME_REPS,
        output_file=BASE_EXPERIMENT_DIRECTORY / RUNNING_TIMES_FILENAME,
    )



def run_file_merges_on_project(repo_name: str, github_user: str) -> int:
    """Run the benchmarks on a single project.

    Args:
        repo_name: Name of the repository.
        github_user: Name of the user/organization that owns the repository.
    Returns:
        The amount of merge scenarios that were used in the benchmark.
    """
    base_merge_dir = BASE_EXPERIMENT_DIRECTORY / github_user / repo_name
    base_merge_dir.mkdir(exist_ok=True, parents=True)
    merge_scenarios_path = base_merge_dir / "merge_scenarios.csv"
    file_merge_output = base_merge_dir / FILE_MERGE_RESULTS_FILENAME
    git_merge_output = base_merge_dir / GIT_MERGE_RESULTS_FILENAME

    if not merge_scenarios_path.exists():
        command.extract_merge_scenarios(
            repo_name=repo_name,
            github_user=github_user,
            non_trivial=True,
            buildable=True,
            testable=False,  # don't want to run tests
            skip_non_content_conflicts=False,  # implied by non_trivial, more eficient to disable
            output_file=merge_scenarios_path,
            timeout=SCENARIO_EXTRACTION_TIME_LIMIT,
        )

    num_merge_scenarios = select_merge_scenarios(
        merge_scenarios_path, limit=MAX_SCENARIOS_PER_PROJECT
    )

    if num_merge_scenarios == 0:
        shutil.rmtree(base_merge_dir)
        return 0

    command.run_file_merges(
        repo_name=repo_name,
        github_user=github_user,
        output_file=file_merge_output,
        merge_scenarios=merge_scenarios_path,
        merge_commands=MERGE_COMMANDS,
        num_merges=None,
        gather_metainfo=True,
        base_merge_dir=BASE_EXPERIMENT_DIRECTORY,
    )

    return num_merge_scenarios


def select_merge_scenarios(merge_scenarios_path: pathlib.Path, limit: int) -> int:
    """Randomly select a maximum of limit merge scenarios and write them back
    into the file.

    If the amount of merge scenarios is already less than the limit, do nothing.

    Returns:
        the amount of merge scenarios
    """
    headers, *scenario_lines = [
        line
        for line in merge_scenarios_path.read_text(
            encoding=sys.getdefaultencoding()
        ).split("\n")
        if line.strip()
    ]
    num_scenarios = len(scenario_lines)
    if num_scenarios <= limit:
        return num_scenarios

    shutil.copy(
        merge_scenarios_path,
        merge_scenarios_path.parent / (merge_scenarios_path.name + ".bak"),
    )
    random_selection = np.random.choice(scenario_lines, size=limit, replace=False)
    lines = "\n".join([headers, *random_selection, ""])
    merge_scenarios_path.write_text(lines, encoding=sys.getdefaultencoding())
    return limit


def setup_logging():
    """Setup the logger to log both to stdout and to a file in the current
    working directory.
    """
    daiquiri.setup(
        level=logging.INFO,
        outputs=(
            daiquiri.output.Stream(
                sys.stdout,
                formatter=daiquiri.formatter.ColorFormatter(
                    fmt="%(color)s[%(levelname)s] %(message)s%(color_stop)s"
                ),
            ),
            daiquiri.output.File(
                filename=str("spork_benchmark.log"),
                formatter=daiquiri.formatter.ColorFormatter(
                    fmt="%(asctime)s [PID %(process)d] [%(levelname)s] "
                    "%(name)s -> %(message)s"
                ),
            ),
        ),
    )


if __name__ == "__main__":
    main()
