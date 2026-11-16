"""Small deterministic checks for metrics, review status, and grouped resampling."""

import math

import numpy as np

from run_experiment import expand_queries, neighbor_scores, ranking_metrics, reviewed_entries, scales
from statistics_plan import intervals_by_project


def main():
    metrics = ranking_metrics(np.array([True, False, True] + [False] * 9), 2)
    assert math.isclose(metrics["ap"], (1 + 2 / 3) / 2)
    assert math.isclose(metrics["ndcg_at_10"], 1.5 / (1 + 1 / math.log2(3)))
    assert metrics["precision_at_5"] == 0.4 and metrics["recall_at_5"] == 1.0
    assert math.isclose(metrics["f1_at_5"], 4 / 7)
    glossary = [{"id": "g1", "terms": ["DAO", "database access object"]}]
    expanded, fired = expand_queries(["use dao", "use daos", "database access object dao"], glossary)
    assert expanded == ["use dao database access object", "use daos", "database access object dao"]
    assert fired == [["g1"], [], []]
    assert reviewed_entries(glossary, {}) is None
    assert reviewed_entries(glossary, {"g1": {"decision": "reject"}}) == []
    assert reviewed_entries(glossary, {"g1": {"decision": "accept"}}) == glossary
    scores = np.array([[0.2, 1.0, 0.6], [0.0, 0.0, 0.0]])
    support = neighbor_scores(scores, [("A", "B"), ("B", "C")], {"A": 0, "B": 1, "C": 2})
    assert np.allclose(support, [[1.0, 0.6, 1.0], [0.0, 0.0, 0.0]])
    assert np.allclose(scales(np.array([[3.0, 3.0], [4.0, 9.0]])), [[0.0, 0.0], [0.0, 1.0]])
    rows = []
    for group, deltas in [("G1", [0.2, 0.2]), ("G2", [-0.1])]:
        for i, delta in enumerate(deltas):
            for method, value in [("no_context", 0.5), ("verified_glossary_and_relations", 0.5 + delta)]:
                rows.append({"project": "toy", "method": method, "requirement": group + str(i),
                             "group": group, "ap": value, "ndcg_at_10": value})
    intervals = intervals_by_project(rows, draws=1000, seed=635)
    assert len(intervals) == 2 and all(row["groups"] == 2 for row in intervals)
    assert all(math.isclose(row["difference"], 0.1) and math.isclose(row["ci_low"], -0.1)
               and math.isclose(row["ci_high"], 0.2) for row in intervals)
    print("Metric, review gate, alias, graph, normalization, and group bootstrap checks passed.")


if __name__ == "__main__":
    main()
