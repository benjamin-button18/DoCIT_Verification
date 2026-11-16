#!/usr/bin/env python3
"""Rerun the Phase 1 experiment and check the archived iTrust pilot results."""

import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
PHASE1 = ROOT / "DoCIT_Phase_1"
PILOT = ROOT / "iTrust_pilot"
DATA = ROOT / "data" / "finegrained-traceability-main.zip"
PUBLISHED = PHASE1 / "outputs" / "preview_run"
FLOAT_FIELDS = {
    "score", "ap", "map", "ndcg_at_10", "precision_at_5", "recall_at_5", "f1_at_5",
    "glossary_fired_rate", "relationship_fired_rate", "top10_changed_rate",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_csv(expected, actual):
    opener = gzip.open if expected.suffix == ".gz" else open
    with opener(expected, "rt", encoding="utf-8", newline="") as left:
        with opener(actual, "rt", encoding="utf-8", newline="") as right:
            saved, rerun = csv.DictReader(left), csv.DictReader(right)
            if saved.fieldnames != rerun.fieldnames:
                raise ValueError(f"Column mismatch in {expected.name}")
            for line, (a, b) in enumerate(zip_longest(saved, rerun), start=2):
                if a is None or b is None:
                    raise ValueError(f"Row count mismatch in {expected.name}")
                for field, value in a.items():
                    if field in FLOAT_FIELDS:
                        equal = math.isclose(float(value), float(b[field]), abs_tol=1e-10, rel_tol=1e-8)
                    else:
                        equal = value == b[field]
                    if not equal:
                        raise ValueError(f"{expected.name}, row {line}, {field}: {value!r} != {b[field]!r}")
    print(f"Matched {expected.name}")


def compare_pilot_aggregates(pilot_zip, dataset_hash):
    def rows(zf, name):
        with zf.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            return list(csv.DictReader(text))

    groups = defaultdict(list)
    with ZipFile(pilot_zip) as zf:
        if json.loads(zf.read("run_manifest.json"))["archive_sha256"] != dataset_hash:
            raise ValueError("iTrust pilot used a different dataset archive")
        for row in rows(zf, "per_requirement_metrics.csv"):
            groups[row["Model"]].append(row)
        for aggregate in rows(zf, "metrics_all_105.csv"):
            name = aggregate[""]
            for metric in ("Precision@5", "Recall@5", "F1@5", "MAP"):
                field = "AP" if metric == "MAP" else metric
                mean = sum(float(row[field]) for row in groups[name]) / len(groups[name])
                if not math.isclose(mean, float(aggregate[metric]), abs_tol=2e-8):
                    raise ValueError(f"iTrust aggregate mismatch for {name}, {metric}")
    print("Matched the saved iTrust aggregate metrics")


def main():
    manifest = json.loads((PUBLISHED / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_hash = sha256(DATA)
    if dataset_hash != manifest["archive_sha256"]:
        raise ValueError("Dataset SHA-256 differs from the published run")
    for name, checksum in manifest["code_sha256"].items():
        if sha256(PHASE1 / name) != checksum:
            raise ValueError(f"Source checksum mismatch for {name}")
    print("Matched the dataset and Phase 1 source checksums", flush=True)

    subprocess.run([sys.executable, str(PHASE1 / "check_pipeline.py")], check=True)
    output = Path(tempfile.mkdtemp(prefix="docit_verification_")) / "phase1_results"
    subprocess.run([
        sys.executable, str(PHASE1 / "run_experiment.py"), str(DATA),
        "--review", str(PHASE1 / "knowledge" / "REVIEW_ME.csv"),
        "--preview", "--output", str(output),
    ], check=True)
    for name in ("dataset_audit.csv", "metrics_by_project.csv", "per_requirement_metrics.csv",
                 "ranked_candidates.csv.gz"):
        check_csv(PUBLISHED / name, output / name)

    subprocess.run([sys.executable, str(ROOT / "analyze_phase1.py"), str(output)], check=True)
    check_csv(ROOT / "phase1_observations.csv", output / "phase1_observations.csv")
    pilot_zip = PILOT / "DoCIT_iTrust_Task1_results.zip"
    compare_pilot_aggregates(pilot_zip, dataset_hash)
    subprocess.run([sys.executable, str(PILOT / "DoCIT_Task2_Verify.py"), str(pilot_zip)], check=True)
    print(f"Verification passed. Fresh Phase 1 results are in {output}")


if __name__ == "__main__":
    main()
