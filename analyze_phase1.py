"""Summarize paired outcomes and context coverage from a Phase 1 run."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


CONDITIONS = ("unverified_glossary", "automatic_graph_only", "unverified_glossary_and_relations_preview")


def summarize(source):
    by_condition = defaultdict(dict)
    with source.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_condition[row["project"], row["method"]][row["requirement"]] = row

    summary = []
    projects = [project for project, method in by_condition if method == "no_context"]
    for project in projects:
        baseline = by_condition[project, "no_context"]
        for method in CONDITIONS:
            variant = by_condition[project, method]
            if variant.keys() != baseline.keys():
                raise ValueError(f"Paired requirements differ for {project}, {method}")
            ap = {key: float(variant[key]["ap"]) - float(row["ap"]) for key, row in baseline.items()}
            ndcg = [float(variant[key]["ndcg_at_10"]) - float(row["ndcg_at_10"])
                    for key, row in baseline.items()]
            summary.append({
                "project": project, "method": method, "linked_requirements": len(baseline),
                "delta_map": f"{mean(ap.values()):.8f}", "delta_ndcg_at_10": f"{mean(ndcg):.8f}",
                "ap_improved": sum(d > 1e-12 for d in ap.values()),
                "ap_tied": sum(abs(d) <= 1e-12 for d in ap.values()),
                "ap_declined": sum(d < -1e-12 for d in ap.values()),
                "glossary_appended": sum(row["glossary_fired"] == "True" for row in variant.values()),
                "positive_relationship_support": sum(row["relationship_fired"] == "True"
                                                     for row in variant.values()),
                "ordered_top10_changed": sum(row["top10_changed"] == "True" for row in variant.values()),
                "largest_ap_gain_requirement": max(ap, key=ap.__getitem__),
                "largest_ap_loss_requirement": min(ap, key=ap.__getitem__),
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = summarize(args.results_dir / "per_requirement_metrics.csv")
    output = args.output or args.results_dir / "phase1_observations.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
