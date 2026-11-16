"""Paired group bootstrap for completed development review batches."""

from collections import defaultdict

import numpy as np


CONTRASTS = (
    ("verified_glossary_and_relations", "no_context"),
    ("verified_glossary_and_relations", "verified_glossary"),
    ("verified_glossary", "unverified_glossary"),
    ("verified_glossary_and_relations", "verified_glossary_and_automatic_graph"),
)


def intervals_by_project(rows, draws=10_000, seed=635):
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["project"], row["method"]][row["requirement"]] = row
    rng = np.random.default_rng(seed)
    intervals = []
    for project in sorted({row["project"] for row in rows}):
        for method, comparator in CONTRASTS:
            treated = by_method.get((project, method))
            control = by_method.get((project, comparator))
            if not treated or not control:
                continue
            if treated.keys() != control.keys():
                raise ValueError(f"{project}: contrast requires matched requirements")
            groups = defaultdict(list)
            for ident, treated_row in sorted(treated.items()):
                control_row = control[ident]
                if control_row["group"] != treated_row["group"]:
                    raise ValueError(f"{project}: conflicting groups for {ident}")
                groups[treated_row["group"]].append([
                    treated_row[key] - control_row[key] for key in ("ap", "ndcg_at_10")
                ])
            if len(groups) < 2:
                continue
            sums = np.array([np.sum(batch, axis=0) for batch in groups.values()])
            counts = np.array([len(batch) for batch in groups.values()])
            samples = rng.integers(len(counts), size=(draws, len(counts)))
            bootstrap = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)[:, None]
            lower, upper = np.quantile(bootstrap, (0.025, 0.975), axis=0)
            estimate = sums.sum(axis=0) / counts.sum()
            for i, metric in enumerate(("map", "ndcg_at_10")):
                intervals.append({
                    "project": project, "method": method, "comparator": comparator, "metric": metric,
                    "difference": estimate[i], "ci_low": lower[i], "ci_high": upper[i],
                    "groups": len(counts), "draws": draws, "seed": seed, "status": "development exploratory",
                })
    return intervals
