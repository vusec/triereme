#!/usr/bin/env python3
from visitor import (
    RemoteVisitor,
    RemoteVisitorBuilder,
    GCSVisitorBuilder,
    LocalVisitorBuilder,
)

import logging
from pathlib import Path
from argparse import ArgumentParser
import subprocess
import json
import csv
from collections import namedtuple
from datetime import timedelta
import re
import datetime
import concurrent.futures
import tempfile
from typing import Optional

import dateutil.parser
from tqdm import tqdm
import pandas as pd

TraceRecord = namedtuple(
    "TraceRecord",
    [
        "file_name",
        "timestamp",
        "total_time_us",
        "trace_time_us",
        "replay_time_us",
        "solve_time_us",
        "export_time_us",
        "solve_call_total_time_us",
        "solve_call_reset_time_us",
        "solve_call_push_time_us",
        "solve_call_assert_time_us",
        "solve_call_check_time_us",
        "solve_call_pop_time_us",
        "solve_call_solve_time_us",
        "solve_call_check_sat",
        "solve_call_check_unsat",
        "solve_call_check_other",
    ],
)

CondRecord = namedtuple(
    "CondRecord",
    [
        "solve_time_us",
        "absolute_path_length",
        "relative_path_length",
        "sat_result",
    ],
)

TimeoutRecord = namedtuple(
    "TimeoutRecord",
    [
        "relative_time_us",
        "timeout_us",
    ],
)

MemoryRecord = namedtuple(
    "TimeoutRecord",
    [
        "relative_time_us",
        "jemalloc_bytes",
        "z3_bytes",
    ],
)


def parse_time_from_entry(log_obj, field: str) -> Optional[int]:
    try:
        seconds = float(log_obj["fields"][field])
        return int(timedelta(seconds=seconds) / timedelta(microseconds=1))
    except KeyError:
        return None


def process_trace_file(
    trace_file_path: Path,
    file_stats_csv_path: Path,
    cond_stats_csv_path: Path,
    timeout_stats_csv_path: Path,
    memory_stats_csv_path: Path,
):
    with tempfile.TemporaryFile(mode="w+") as temp_file:
        subprocess.run(
            ["gunzip", "--stdout", f"{trace_file_path}"],
            stdout=temp_file,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        temp_file.seek(0)

        with open(file_stats_csv_path, "w", newline="") as file_stats_file, open(
            cond_stats_csv_path, "w", newline=""
        ) as cond_stats_file, open(
            timeout_stats_csv_path, "w", newline=""
        ) as timeout_stats_file, open(
            memory_stats_csv_path, "w", newline=""
        ) as memory_stats_file:
            file_csv_writer = csv.DictWriter(
                file_stats_file, fieldnames=TraceRecord._fields
            )
            file_csv_writer.writeheader()

            cond_csv_writer = csv.DictWriter(
                cond_stats_file, fieldnames=CondRecord._fields
            )
            cond_csv_writer.writeheader()

            timeout_csv_writer = csv.DictWriter(
                timeout_stats_file, fieldnames=TimeoutRecord._fields
            )
            timeout_csv_writer.writeheader()

            memory_csv_writer = csv.DictWriter(
                memory_stats_file, fieldnames=MemoryRecord._fields
            )
            memory_csv_writer.writeheader()

            experiment_start = None
            current_file_name = None
            current_test_case_start = None
            current_trace_time = None
            current_replay_time = None
            current_solve_time = None
            current_export_time = None
            current_test_input_time = None

            current_solve_call_total_time = None
            current_solve_call_reset_time = None
            current_solve_call_push_time = None
            current_solve_call_assert_time = None
            current_solve_call_check_time = None
            current_solve_call_pop_time = None
            current_solve_call_solve_time = None
            current_solve_call_sat = None
            current_solve_call_unsat = None
            current_solve_call_other = None

            for log_line in temp_file:
                log_obj = json.loads(log_line)

                message = log_obj["fields"]["message"]
                input_match = re.fullmatch("Running on input (.*)", message)
                timeout_match = re.fullmatch("Setting timeout at ([0-9]+) ms", message)

                if message == "trace timing":
                    current_trace_time = parse_time_from_entry(log_obj, "tracing_time")
                elif message == "replay time":
                    current_replay_time = parse_time_from_entry(log_obj, "replay_time")
                elif message == "total solve time":
                    current_solve_time = parse_time_from_entry(log_obj, "solve_time")
                elif message == "export time":
                    current_export_time = parse_time_from_entry(log_obj, "export_time")
                elif message == "test input time":
                    current_test_input_time = int(
                        log_obj["fields"]["test_input_time_us"]
                    )
                elif message == "solve call":
                    current_solve_call_total_time = parse_time_from_entry(
                        log_obj, "total_time"
                    )

                    current_solve_call_reset_time = parse_time_from_entry(
                        log_obj, "total_reset_time"
                    )
                    current_solve_call_push_time = parse_time_from_entry(
                        log_obj, "total_push_time"
                    )
                    current_solve_call_assert_time = parse_time_from_entry(
                        log_obj, "total_assert_time"
                    )
                    current_solve_call_check_time = parse_time_from_entry(
                        log_obj, "total_check_time"
                    )
                    current_solve_call_pop_time = parse_time_from_entry(
                        log_obj, "total_pop_time"
                    )

                    current_solve_call_solve_time = parse_time_from_entry(
                        log_obj, "total_solve_time"
                    )

                    current_solve_call_sat = log_obj["fields"]["solver_check_sat"]
                    current_solve_call_unsat = log_obj["fields"]["solver_check_unsat"]
                    current_solve_call_other = log_obj["fields"]["solver_check_other"]
                elif input_match is not None:
                    new_test_case_start = dateutil.parser.parse(log_obj["timestamp"])

                    if (
                        current_file_name is not None
                        and current_trace_time is not None
                        and current_replay_time is not None
                        and current_solve_time is not None
                        and current_export_time is not None
                    ):
                        if current_test_input_time is not None:
                            total_time_us = current_test_input_time
                        else:
                            logging.warning("test case time not present, approximating")
                            total_time_us = int(
                                (new_test_case_start - current_test_case_start)
                                / datetime.timedelta(microseconds=1)
                            )

                        record = TraceRecord(
                            current_file_name,
                            current_test_case_start,
                            total_time_us,
                            current_trace_time,
                            current_replay_time,
                            current_solve_time,
                            current_export_time,
                            current_solve_call_total_time,
                            current_solve_call_reset_time,
                            current_solve_call_push_time,
                            current_solve_call_assert_time,
                            current_solve_call_check_time,
                            current_solve_call_pop_time,
                            current_solve_call_solve_time,
                            current_solve_call_sat,
                            current_solve_call_unsat,
                            current_solve_call_other,
                        )
                        file_csv_writer.writerow(record._asdict())
                    elif (
                        current_trace_time is not None
                        and current_replay_time is None
                        and current_solve_time is None
                        and current_export_time is not None
                    ):
                        logging.warning("Possible backend crash detected")

                    current_file_name = input_match.group(1)
                    current_test_case_start = new_test_case_start
                    current_trace_time = None
                    current_replay_time = None
                    current_solve_time = None
                    current_export_time = None
                    current_test_input_time = None

                    current_solve_call_total_time = None
                    current_solve_call_reset_time = None
                    current_solve_call_push_time = None
                    current_solve_call_assert_time = None
                    current_solve_call_check_time = None
                    current_solve_call_pop_time = None
                    current_solve_call_solve_time = None
                    current_solve_call_sat = None
                    current_solve_call_unsat = None
                    current_solve_call_other = None

                elif message == "solver check call":
                    solve_time = parse_time_from_entry(log_obj, "solve_time")
                    if solve_time is None:
                        solve_time = parse_time_from_entry(log_obj, "check_time")
                    record = CondRecord(
                        solve_time,
                        log_obj["fields"]["absolute_path_length"],
                        log_obj["fields"]["relative_path_length"],
                        log_obj["fields"]["sat_result"],
                    )
                    cond_csv_writer.writerow(record._asdict())
                elif message == "runtime constructed":
                    if experiment_start is None:
                        experiment_start = dateutil.parser.parse(log_obj["timestamp"])
                elif timeout_match is not None:
                    timeout_timestamp = dateutil.parser.parse(log_obj["timestamp"])
                    timeout = timedelta(milliseconds=int(timeout_match.group(1)))
                    timeout_record = TimeoutRecord(
                        (timeout_timestamp - experiment_start)
                        / timedelta(microseconds=1),
                        timeout / timedelta(microseconds=1),
                    )

                    timeout_csv_writer.writerow(timeout_record._asdict())
                elif message == "memory usage":
                    memory_timestamp = dateutil.parser.parse(log_obj["timestamp"])
                    jemalloc_bytes = log_obj["fields"]["rust"]
                    z3_bytes = log_obj["fields"]["z3"]

                    if experiment_start is None:
                        experiment_start = memory_timestamp

                    memory_record = MemoryRecord(
                        (memory_timestamp - experiment_start)
                        / timedelta(microseconds=1),
                        jemalloc_bytes,
                        z3_bytes,
                    )

                    memory_csv_writer.writerow(memory_record._asdict())


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


TRACE_ARCHIVE_PATH = "corpus/mine/trace.json.gz"


def extract_stats(visitor: RemoteVisitor, trial_dir: Path, tmp_dir: Path):
    corpus_dir = trial_dir / "corpus"
    corpus_files = visitor.list_files(corpus_dir)
    archive_path = tmp_dir / "corpus.tar.gz"
    for corpus in sorted(corpus_files, reverse=True):
        visitor.retrieve_file(corpus, archive_path)
        if is_file_in_archive(archive_path, TRACE_ARCHIVE_PATH):
            break
        archive_path.unlink()
    if not archive_path.is_file():
        raise Exception(f"FAILED: {trial_dir}")

    trace_file_path = tmp_dir / "trace.json.gz"
    try:
        subprocess.run(
            [
                "tar",
                "--extract",
                f"--file={archive_path}",
                f"--directory={tmp_dir}",
                f"--transform=s/.*/{trace_file_path.name}/",
                TRACE_ARCHIVE_PATH,
            ],
            check=True,
        )
    except:
        tqdm.write(f"FAILED: {archive_path}")
        raise

    archive_path.unlink()

    file_stats_csv_name = tmp_dir / "file_stats.csv"
    cond_stats_csv_name = tmp_dir / "cond_stats.csv"
    timeout_stats_csv_name = tmp_dir / "timeout_stats.csv"
    memory_stats_csv_name = tmp_dir / "memory_stats.csv"

    process_trace_file(
        trace_file_path,
        file_stats_csv_name,
        cond_stats_csv_name,
        timeout_stats_csv_name,
        memory_stats_csv_name,
    )

    file_stats = pd.read_csv(file_stats_csv_name)
    cond_stats = pd.read_csv(cond_stats_csv_name)
    timeout_stats = pd.read_csv(timeout_stats_csv_name)
    memory_stats = pd.read_csv(memory_stats_csv_name)

    return (file_stats, cond_stats, timeout_stats, memory_stats)


def serialize_stats(stats: pd.DataFrame, benchmark, fuzzer, trial_id, cache_file: Path):
    stats["trial_id"] = trial_id
    stats["benchmark"] = benchmark
    stats["fuzzer"] = fuzzer
    stats.to_csv(cache_file, index=False)


def process_trial(
    visitor_builder: RemoteVisitorBuilder,
    trial_dir: Path,
    benchmark,
    fuzzer,
    trial_id,
    cache_dir: Path,
):
    tqdm.write(f"Processing: {cache_dir}")
    visitor = visitor_builder.build()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        file_stats, cond_stats, timeout_stats, memory_stats = extract_stats(
            visitor, trial_dir, tmp_dir
        )

    file_stats_csv = cache_dir / "file_stats.csv.gz"
    serialize_stats(file_stats, benchmark, fuzzer, trial_id, file_stats_csv)

    cond_stats_csv = cache_dir / "cond_stats.csv.gz"
    cond_stats = cond_stats.sample(frac=1 / 100)
    serialize_stats(cond_stats, benchmark, fuzzer, trial_id, cond_stats_csv)

    timeout_stats_csv = cache_dir / "timeout_stats.csv.gz"
    serialize_stats(timeout_stats, benchmark, fuzzer, trial_id, timeout_stats_csv)

    memory_stats_csv = cache_dir / "memory_stats.csv.gz"
    serialize_stats(memory_stats, benchmark, fuzzer, trial_id, memory_stats_csv)

    return (file_stats_csv, cond_stats_csv, timeout_stats_csv, memory_stats_csv)


def resample_trial_memory_dataframe(memory_df: pd.DataFrame):
    first_row = memory_df.iloc[0].copy()
    first_row["relative_time_us"] = 0.0
    memory_df = pd.concat([first_row.to_frame().T, memory_df])

    memory_df["relative_time"] = pd.to_timedelta(
        memory_df["relative_time_us"], unit="micros"
    )
    memory_df.set_index("relative_time", inplace=True)

    resampled = memory_df.resample("15Min", origin="epoch").max()
    resampled.reset_index(inplace=True)

    resampled["relative_time_us"] = resampled["relative_time"] / timedelta(
        microseconds=1
    )
    del resampled["relative_time"]

    memory_df.reset_index(inplace=True)
    del memory_df["relative_time"]

    return resampled


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
    fuzzers &= {"symcc_linear_single", "symcc_trie_single"}

    file_stats = pd.DataFrame()
    cond_stats = pd.DataFrame()
    timeout_stats = pd.DataFrame()
    memory_stats = pd.DataFrame()

    future_to_archive = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for benchmark in sorted(benchmarks):
            for fuzzer in sorted(fuzzers):
                pair_dir = experiment_folders_dir / f"{benchmark}-{fuzzer}"
                trial_dirs = visitor.list_subdirs(pair_dir)
                for trial_dir in sorted(trial_dirs):
                    trial_id = int(trial_dir.name[len("trial-") :])
                    trial_cache_dir = cache_dir / f"{benchmark}-{fuzzer}-{trial_id}"

                    file_stats_csv = trial_cache_dir / "file_stats.csv.gz"
                    cond_stats_csv = trial_cache_dir / "cond_stats.csv.gz"
                    timeout_stats_csv = trial_cache_dir / "timeout_stats.csv.gz"
                    memory_stats_csv = trial_cache_dir / "memory_stats.csv.gz"
                    if (
                        not file_stats_csv.is_file()
                        or not cond_stats_csv.is_file()
                        or not timeout_stats_csv.is_file()
                        or not memory_stats_csv.is_file()
                    ):
                        trial_cache_dir.mkdir(exist_ok=True)
                        future = executor.submit(
                            process_trial,
                            visitor_builder,
                            trial_dir,
                            benchmark,
                            fuzzer,
                            trial_id,
                            trial_cache_dir,
                        )
                        future_to_archive[future] = trial_cache_dir
                    else:
                        print(f"Loading from: {trial_cache_dir}")

                        if "file" not in args.skip:
                            file_stats_trial = pd.read_csv(file_stats_csv)
                            file_stats = pd.concat(
                                [file_stats, file_stats_trial], ignore_index=True
                            )

                        if "cond" not in args.skip:
                            cond_stats_trial = pd.read_csv(cond_stats_csv)
                            cond_stats = pd.concat(
                                [cond_stats, cond_stats_trial], ignore_index=True
                            )

                        if "timeout" not in args.skip:
                            timeout_stats_trial = pd.read_csv(timeout_stats_csv)
                            timeout_stats = pd.concat(
                                [timeout_stats, timeout_stats_trial], ignore_index=True
                            )

                        if "memory" not in args.skip:
                            memory_stats_trial = pd.read_csv(memory_stats_csv)
                            memory_stats_trial = resample_trial_memory_dataframe(
                                memory_stats_trial
                            )
                            memory_stats = pd.concat(
                                [memory_stats, memory_stats_trial], ignore_index=True
                            )

        for future in tqdm(
            concurrent.futures.as_completed(future_to_archive),
            total=len(future_to_archive),
        ):
            (
                file_stats_csv,
                cond_stats_csv,
                timeout_stats_csv,
                memory_stats_csv,
            ) = future.result()  # Raise exception, if any.
            tqdm.write(f"Done: {future_to_archive[future]}")

            if "file" not in args.skip:
                file_stats_trial = pd.read_csv(file_stats_csv)
                file_stats = pd.concat(
                    [file_stats, file_stats_trial], ignore_index=True
                )

            if "cond" not in args.skip:
                cond_stats_trial = pd.read_csv(cond_stats_csv)
                cond_stats = pd.concat(
                    [cond_stats, cond_stats_trial], ignore_index=True
                )

            if "timeout" not in args.skip:
                timeout_stats_trial = pd.read_csv(timeout_stats_csv)
                timeout_stats = pd.concat(
                    [timeout_stats, timeout_stats_trial], ignore_index=True
                )

            if "memory" not in args.skip:
                memory_stats_trial = pd.read_csv(memory_stats_csv)
                memory_stats_trial = resample_trial_memory_dataframe(memory_stats_trial)
                memory_stats = pd.concat(
                    [memory_stats, memory_stats_trial], ignore_index=True
                )

    if "file" not in args.skip:
        file_stats_output_path: Path = (
            output_dir / f"{experiment_name}-symcc_trace-file_stats.csv.gz"
        )
        file_stats.to_csv(file_stats_output_path, index=False)
        print(f"output: {file_stats_output_path}")

    if "cond" not in args.skip:
        cond_stats_output_path: Path = (
            output_dir / f"{experiment_name}-symcc_trace-cond_stats-sampled.csv.gz"
        )
        cond_stats.to_csv(cond_stats_output_path, index=False)
        print(f"output: {cond_stats_output_path}")

    if "timeout" not in args.skip:
        timeout_stats_output_path: Path = (
            output_dir / f"{experiment_name}-symcc_trace-timeout_stats.csv.gz"
        )
        timeout_stats.to_csv(timeout_stats_output_path, index=False)
        print(f"output: {timeout_stats_output_path}")

    if "memory" not in args.skip:
        memory_stats_output_path: Path = (
            output_dir / f"{experiment_name}-symcc_trace-memory_stats.csv.gz"
        )
        memory_stats.to_csv(memory_stats_output_path, index=False)
        print(f"output: {memory_stats_output_path}")


def main(args):
    if args.source.startswith("gs://"):
        bucket_name = args.source.removeprefix("gs://")
        visitor_builder = GCSVisitorBuilder(bucket_name)
    else:
        visitor_builder = LocalVisitorBuilder(args.source)
    process_experiment(visitor_builder, args.experiment_name, args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("experiment_name")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--skip",
        choices=["file", "cond", "timeout", "memory"],
        default=["timeout", "memory"],
        nargs="*",
    )
    args = parser.parse_args()
    main(args)
