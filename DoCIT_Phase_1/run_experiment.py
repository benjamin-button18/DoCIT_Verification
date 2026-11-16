"""Run the development-only retrieval comparison; keep review status in every output."""

import argparse
import csv
import gzip
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import javalang
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import audit_row, benchmark_projects, digest_file
from java_graph import static_graph
from statistics_plan import intervals_by_project


A, BETA = 0.1, 0.1
HERE = Path(__file__).resolve().parent
METRICS = ("ap", "ndcg_at_10", "precision_at_5", "recall_at_5", "f1_at_5")


def tokens(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def scales(values):
    low = values.min(axis=1, keepdims=True)
    spread = values.max(axis=1, keepdims=True) - low
    return np.divide(values - low, spread, out=np.zeros_like(values), where=spread > 0)


def expand_queries(queries, entries):
    groups = [(entry["id"], sorted(set(map(tokens, entry["terms"])))) for entry in entries]
    expanded, matched = [], []
    for query in queries:
        padded = f" {query} "
        additions = set()
        ids = []
        for ident, terms in groups:
            if any(f" {term} " in padded for term in terms):
                missing = {term for term in terms if f" {term} " not in padded}
                if missing:
                    additions.update(missing)
                    ids.append(ident)
        expanded.append(" ".join((query, *sorted(additions))))
        matched.append(ids)
    return expanded, matched


def neighbor_scores(scores, pairs, positions):
    support = np.zeros_like(scores)
    for left, right in sorted(set(pairs)):
        i, j = positions[left], positions[right]
        support[:, i] = np.maximum(support[:, i], scores[:, j])
        support[:, j] = np.maximum(support[:, j], scores[:, i])
    return support


def pairs_from(entries):
    return [(entry["source"], entry["target"]) for entry in entries]


def reviewed_entries(entries, decisions):
    valid = {"accept", "reject"}
    if not entries or any(decisions.get(entry["id"], {}).get("decision") not in valid for entry in entries):
        return None
    return [entry for entry in entries if decisions[entry["id"]]["decision"] == "accept"]


def load_knowledge(draft_path, review_path, projects):
    drafts = json.loads(draft_path.read_text(encoding="utf-8"))
    files = {project.name: project.code for project in projects}
    known = set()
    for entry in drafts:
        key = entry["id"]
        if key in known:
            raise ValueError(f"Duplicate knowledge ID {key}")
        known.add(key)
        source = files[entry["project"]]
        lines = source[entry["source"]].splitlines()
        first, last = entry["lines"]
        if not 1 <= first <= last <= len(lines) or entry["quote"] != "\n".join(lines[first - 1:last]):
            raise ValueError(f"{key}: evidence span differs from the benchmark")
        if entry["kind"] == "glossary":
            if len({tokens(term) for term in entry["terms"] if tokens(term)}) < 2:
                raise ValueError(f"{key}: at least two different glossary terms are required")
        elif entry["kind"] == "relationship":
            if (entry["target"] not in source or entry["target"] == entry["source"]
                    or entry["relation"] not in {"call", "inheritance", "import"}):
                raise ValueError(f"{key}: invalid code relationship")
        else:
            raise ValueError(f"{key}: unknown entry kind")
    decisions = {}
    if review_path:
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["id"]
                if key not in known or key in decisions:
                    raise ValueError(f"Unknown or duplicate review ID {key}")
                row["decision"] = row["decision"].strip().lower()
                if row["decision"] not in {"", "accept", "reject"}:
                    raise ValueError(f"{key}: use accept, reject, or leave blank")
                if row["decision"] and not all(row[field].strip() for field in ("reviewer", "reason")):
                    raise ValueError(f"{key}: a decision requires reviewer and reason")
                decisions[key] = row
    return drafts, decisions


def build_rankers(project, entries, accepted, preview):
    query_ids, files = sorted(project.requirements), sorted(project.code)
    positions = {name: i for i, name in enumerate(files)}
    queries = [tokens(project.requirements[key]) for key in query_ids]
    corpus = [tokens(f"{key}\n{project.code[key]}") for key in files]
    vectorizer = TfidfVectorizer(stop_words=project.stop_words, ngram_range=(1, 2), sublinear_tf=True)
    target_vectors = vectorizer.fit_transform(corpus)

    def similarity(query_text):
        return scales((vectorizer.transform(query_text) @ target_vectors.T).toarray())

    def glossary(entries):
        expanded, matches = expand_queries(queries, [e for e in entries if e["kind"] == "glossary"])
        return similarity(expanded), matches

    base = similarity(queries)
    length_only = np.broadcast_to([len(tokens(project.code[key]).split()) for key in files], base.shape)
    edges, graph_audit = static_graph(project.code)
    automatic = [(edge.source, edge.target) for edge in edges]
    glossary_draft, draft_matches = glossary(entries)
    graph_only = neighbor_scores(base, automatic, positions)
    rankers = {
        "no_context": (base, None, None),
        "length_only": (length_only, None, None),
        "unverified_glossary": ((1 - A) * base + A * glossary_draft, draft_matches, None),
    }
    if accepted is None:
        rankers["automatic_graph_only"] = ((1 - BETA) * base + BETA * graph_only, None, graph_only)
    else:
        verified, matches = glossary(accepted)
        curated = neighbor_scores(verified, pairs_from(e for e in accepted if e["kind"] == "relationship"), positions)
        static = neighbor_scores(verified, automatic, positions)
        common = (1 - A - BETA) * base + A * verified
        rankers["verified_glossary"] = ((1 - A) * base + A * verified, matches, None)
        rankers["verified_glossary_and_relations"] = (common + BETA * curated, matches, curated)
        rankers["verified_glossary_and_automatic_graph"] = (common + BETA * static, matches, static)
    if preview:
        draft_relations = [entry for entry in entries if entry["kind"] == "relationship"]
        draft_support = neighbor_scores(glossary_draft, pairs_from(draft_relations), positions)
        rankers["unverified_glossary_and_relations_preview"] = (
            (1 - A - BETA) * base + A * glossary_draft + BETA * draft_support,
            draft_matches,
            draft_support,
        )
    return query_ids, files, rankers, graph_audit


def ranking_metrics(hit, reference_count):
    ranks = np.flatnonzero(hit) + 1
    ap = np.sum(np.arange(1, len(ranks) + 1) / ranks) / reference_count
    discount = 1 / np.log2(np.arange(2, 12))
    dcg = hit[:10] @ discount[:len(hit[:10])]
    ndcg = dcg / discount[:min(reference_count, 10)].sum()
    n_top = int(hit[:5].sum())
    precision, recall = n_top / 5, n_top / reference_count
    f1 = 2 * precision * recall / (precision + recall) if n_top else 0.0
    return dict(zip(METRICS, map(float, (ap, ndcg, precision, recall, f1))))


def score_project(project, rankers, query_ids, files, ranking_writer):
    base_order = np.argsort(-rankers["no_context"][0], axis=1, kind="stable")
    positions = {target: i for i, target in enumerate(files)}
    results, summary = [], []
    for method, (scores, matches, relation) in rankers.items():
        order = np.argsort(-scores, axis=1, kind="stable")
        rows = []
        for row, requirement in enumerate(query_ids):
            references = project.references.get(requirement, set())
            expected = {positions[name] for name in references}
            hits = np.array([index in expected for index in order[row]])
            for rank, (index, hit) in enumerate(zip(order[row], hits), 1):
                ranking_writer.writerow((project.name, method, requirement, rank, files[index],
                                         f"{scores[row, index]:.10g}", int(hit)))
            if references:
                rows.append({
                    "project": project.name,
                    "method": method,
                    "requirement": requirement,
                    "group": project.groups[requirement],
                    "recorded_links": len(references),
                    **ranking_metrics(hits, len(references)),
                    "glossary_fired": bool(matches and matches[row]),
                    "matched_glossary_ids": ";".join(matches[row]) if matches else "",
                    "relationship_fired": bool(relation is not None and np.any(relation[row] > 0)),
                    "top10_changed": not np.array_equal(order[row, :10], base_order[row, :10]),
                })
        results.extend(rows)
        summary.append({
            "project": project.name,
            "method": method,
            "linked_requirements": len(rows),
            "map": np.mean([row["ap"] for row in rows]),
            **{key: np.mean([row[key] for row in rows]) for key in METRICS[1:]},
            **{key + "_rate": np.mean([row[key] for row in rows]) for key in
               ("glossary_fired", "relationship_fired", "top10_changed")},
        })
    return results, summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="finegrained-traceability-main.zip")
    parser.add_argument("--drafts", type=Path, default=HERE / "knowledge" / "draft_entries.json")
    parser.add_argument("--review", type=Path, help="review sheet with explicit reviewer decisions")
    parser.add_argument("--preview", action="store_true", help="include the unverified full-layer diagnostic")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    projects = list(benchmark_projects(args.archive))
    entries, decisions = load_knowledge(args.drafts, args.review, projects)
    holdout_audit = [audit_row(project, "provisional holdout; structural audit only")
                     for project in benchmark_projects(args.archive, audit_holdout_only=True)]
    args.output.mkdir(parents=True, exist_ok=False)
    metric_rows, summaries, audit = [], [], []
    manifest = {
        "stage": "development_exploratory",
        "archive_sha256": digest_file(args.archive),
        "draft_sha256": digest_file(args.drafts),
        "review_sha256": digest_file(args.review) if args.review else None,
        "code_sha256": {name: digest_file(HERE / name) for name in
                        ("run_experiment.py", "datasets.py", "java_graph.py", "statistics_plan.py")},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "javalang": javalang.__version__},
        "weights": {"glossary": A, "relation": BETA, "status": "fixed preview; not tuned or frozen"},
        "projects": {},
    }
    with gzip.open(args.output / "ranked_candidates.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("project", "method", "requirement", "rank", "target", "score", "recorded_link"))
        for project in projects:
            audit.append(audit_row(project, "development"))
            project_entries = [e for e in entries if e["project"] == project.name]
            accepted = reviewed_entries(project_entries, decisions)
            qids, files, rankers, graph_info = build_rankers(project, project_entries, accepted, args.preview)
            per_query, aggregate = score_project(project, rankers, qids, files, writer)
            metric_rows.extend(per_query)
            summaries.extend(aggregate)
            manifest["projects"][project.name] = {
                "draft_entries": len(project_entries),
                "verified_comparison": accepted is not None,
                "accepted_entries": len(accepted) if accepted is not None else 0,
                "static_graph": graph_info,
            }
    write_csv(args.output / "dataset_audit.csv", audit + holdout_audit)
    write_csv(args.output / "metrics_by_project.csv", summaries)
    write_csv(args.output / "per_requirement_metrics.csv", metric_rows)
    intervals = intervals_by_project(metric_rows)
    if intervals:
        write_csv(args.output / "development_group_intervals.csv", intervals)
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summaries)} development comparisons to {args.output}")


if __name__ == "__main__":
    main()
