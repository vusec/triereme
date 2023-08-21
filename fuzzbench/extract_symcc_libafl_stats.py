#!/usr/bin/env python3
from visitor import (
    RemoteVisitor,
    RemoteVisitorBuilder,
    GCSVisitorBuilder,
    LocalVisitorBuilder,
)

from pathlib import Path
from argparse import ArgumentParser
import subprocess
import tempfile
import io
import re
from collections import namedtuple
import csv
import concurrent.futures
from datetime import datetime

import pandas as pd
from tqdm import tqdm


def is_file_in_archive(archive_path: Path, file_path: str):
    tar_list = subprocess.run(
        [
            "tar",
            "--list",
            f"--file={archive_path}",
            file_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tar_list.returncode == 0


def read_truncated_gzip(path: Path):
    result = subprocess.run(
        ["gunzip", "-c", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return result.stdout


QSYM_STATS_ARCHIVE_PATH = "corpus/symcc/trace.csv.gz"


def extract_qsym_stats(visitor: RemoteVisitor, trial_dir: Path, tmp_dir: Path):
    corpus_dir = trial_dir / "corpus"
    corpus_files = visitor.list_files(corpus_dir)
    archive_path = tmp_dir / "corpus.tar.gz"
    for corpus in sorted(corpus_files, reverse=True):
        visitor.retrieve_file(corpus, archive_path)
        if is_file_in_archive(archive_path, QSYM_STATS_ARCHIVE_PATH):
            break
        archive_path.unlink()
    if not archive_path.is_file():
        raise Exception(f"FAILED: {trial_dir}")

    qsym_stats_path = tmp_dir / "qsym_stats.csv.gz"
    try:
        subprocess.run(
            [
                "tar",
                "--extract",
                f"--file={archive_path}",
                f"--directory={tmp_dir}",
                f"--transform=s/.*/{qsym_stats_path.name}/",
                QSYM_STATS_ARCHIVE_PATH,
            ],
            check=True,
        )
    except:
        tqdm.write(f"FAILED: {archive_path}")
        raise

    archive_path.unlink()

    qsym_stats_data = read_truncated_gzip(qsym_stats_path).decode()
    return pd.read_csv(io.StringIO(qsym_stats_data), index_col="testcase")


PREFIX_LEN = len("[2022-12-02T11:34:10Z INFO  symcc_fuzzing_helper] ")

RUNNING_ON = re.compile("Running on input (.*)")
GENERATED_REGEX = re.compile(
    r"Generated ([0-9]+) test cases \(([0-9]+) new, export time: ([0-9]+) us, "
    r"symcc time: ([0-9]+) us, solver time: (.*) us\)"
)
TEST_INPUT_REGEX = re.compile("test input time: ([0-9]+) us")

TraceRecord = namedtuple(
    "TraceRecord",
    [
        "file_name",
        "timestamp",
        "total_time_us",
        "symcc_time_us",
        "solve_time_us",
        "export_time_us",
    ],
)


def process_trace_file(
    trace_file_path: Path,
    file_stats_csv_path: Path,
):
    with open(trace_file_path) as trace_file, open(
        file_stats_csv_path, "w"
    ) as file_stats_file:
        file_csv_writer = csv.DictWriter(
            file_stats_file, fieldnames=TraceRecord._fields
        )
        file_csv_writer.writeheader()

        current_file_name = None
        current_timestamp = None
        current_total_time = None
        current_symcc_time = None
        current_solve_time = None
        current_export_time = None

        for log_line in trace_file:
            log_tokens = log_line.split()
            log_line = log_line[PREFIX_LEN:]

            running_on_match = RUNNING_ON.match(log_line)
            generated_match = GENERATED_REGEX.match(log_line)
            test_input_match = TEST_INPUT_REGEX.match(log_line)

            if running_on_match is not None:
                if current_file_name is not None:
                    record = TraceRecord(
                        current_file_name,
                        current_timestamp,
                        current_total_time,
                        current_symcc_time,
                        current_solve_time,
                        current_export_time,
                    )
                    file_csv_writer.writerow(record._asdict())
                current_file_name = running_on_match.group(1)
                current_timestamp = datetime.fromisoformat(log_tokens[0][1:-1])
                current_total_time = None
                current_symcc_time = None
                current_solve_time = None
                current_export_time = None

            elif generated_match is not None:
                current_export_time = generated_match.group(3)
                current_symcc_time = generated_match.group(4)
                solve_time = generated_match.group(5)
                if solve_time != "unknown":
                    current_solve_time = solve_time
            elif test_input_match is not None:
                current_total_time = test_input_match.group(1)


def extract_file_stats(visitor: RemoteVisitor, trial_dir: Path, tmp_dir: Path):
    log_bucket_path = trial_dir / "results/fuzzer-log.txt"
    log_local_path = tmp_dir / "fuzzer-log.txt"
    visitor.retrieve_file(log_bucket_path, log_local_path)

    file_stats_path = tmp_dir / "file_stats.csv"
    process_trace_file(log_local_path, file_stats_path)

    return pd.read_csv(file_stats_path)


def process_trial(
    visitor_builder: RemoteVisitorBuilder,
    trial_dir: Path,
    benchmark,
    fuzzer,
    trial_id,
    cache_file: Path,
):
    visitor = visitor_builder.build()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        qsym_stats = extract_qsym_stats(visitor, trial_dir, tmp_dir)
        file_stats = extract_file_stats(visitor, trial_dir, tmp_dir)

    file_stats["file_name"] = file_stats["file_name"].str.removeprefix(
        "/out/corpus/afl/queue/"
    )
    file_stats.set_index("file_name", inplace=True)

    stats = file_stats.join(qsym_stats, validate="1:1", rsuffix="_qsym")
    stats.reset_index(inplace=True)

    stats["trial_id"] = trial_id
    stats["benchmark"] = benchmark
    stats["fuzzer"] = fuzzer
    stats.to_csv(cache_file, index=False)


def process_experiment(
    visitor_builder: RemoteVisitorBuilder, experiment_name: str, output_dir: Path
):
    visitor = visitor_builder.build()

    output_dir.mkdir(exist_ok=True)
    cache_dir: Path = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    experiment_folders_dir = Path(experiment_name) / "experiment-folders"
    pair_dirs = visitor.list_subdirs(experiment_folders_dir)

    benchmarks = set()
    fuzzers = set()
    for pair_path in pair_dirs:
        benchmark, fuzzer = pair_path.name.rsplit("-", 1)
        benchmarks.add(benchmark)
        fuzzers.add(fuzzer)

    print(f"benchmarks: {benchmarks}")
    print(f"fuzzers: {fuzzers}")
    fuzzers = {"symcc_libafl_single"}

    data = pd.DataFrame()

    future_to_archive = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for benchmark in sorted(benchmarks):
            for fuzzer in sorted(fuzzers):
                pair_dir = experiment_folders_dir / f"{benchmark}-{fuzzer}"
                trial_dirs = visitor.list_subdirs(pair_dir)
                for trial_dir in sorted(trial_dirs):
                    trial_id = int(trial_dir.name[len("trial-") :])
                    cache_file = cache_dir / f"{benchmark}-{fuzzer}-{trial_id}.csv.gz"

                    if not cache_file.is_file():
                        future = executor.submit(
                            process_trial,
                            visitor_builder,
                            trial_dir,
                            benchmark,
                            fuzzer,
                            trial_id,
                            cache_file,
                        )
                        future_to_archive[future] = cache_file
                    else:
                        stats = pd.read_csv(cache_file)
                        data = pd.concat([data, stats], ignore_index=True)

        for future in tqdm(
            concurrent.futures.as_completed(future_to_archive),
            total=len(future_to_archive),
        ):
            future.result()  # Raise exception, if any.
            tqdm.write(f"Done: {future_to_archive[future]}")

            cache_file = future_to_archive[future]
            stats = pd.read_csv(cache_file)
            data = pd.concat([data, stats], ignore_index=True)

    output_path: Path = output_dir / f"{experiment_name}-symcc_libafl_stats.csv.gz"
    data.to_csv(output_path, index=False)
    print(f"output: {output_path}")


def main(args):
    if args.source.startswith("gs://"):
        bucket_name = args.source.removeprefix("gs://")
        visitor_builder = GCSVisitorBuilder(bucket_name)
    else:
        visitor_builder = LocalVisitorBuilder(Path(args.source))
    process_experiment(visitor_builder, args.experiment_name, args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("experiment_name")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args)
