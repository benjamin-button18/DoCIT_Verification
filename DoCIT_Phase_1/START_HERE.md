# DoCIT experiment · Phase 1

This package runs a first development pass for the proposed study. The reference matrices were checked for valid file IDs and duplicate pairs. The 22 glossary and relationship proposals are awaiting human review. The run uses a small starter batch and provisional weights of 0.10 for each context component.

## Results from the development preview

| Project | Requirements | Candidate files | Reference links | Requirements scored | No-context MAP | No-context nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| iTrust | 131 | 226 | 286 | 105 | 0.2933 | 0.3680 |
| eTour | 58 | 116 | 308 | 57 | 0.4047 | 0.4762 |
| SMOS | 67 | 100 | 1,044 | 67 | 0.3745 | 0.4501 |
| eANCI, provisional public holdout | 139 | 55 | 567 | Reserved | | |

The export contains five conditions for each development project. They are no context, length only, unverified glossary, automatic graph alone, and unverified glossary plus draft relationships. It includes complete rankings, metrics per requirement, and project-level summaries. The code-only TF-IDF fit follows the new proposal and differs from the full-corpus fit used for the V2.1 pilot.

The draft glossary appended terms for 6 of 105 iTrust queries, 17 of 57 eTour queries, and 5 of 67 SMOS queries. It changed the ordered top ten for 2 queries in each project. These early coverage counts describe a small draft batch. The verified-context comparison will use the human review decisions. An appended term can be out of vocabulary, and a nonzero relation score can leave the ordering unchanged.

### Data and parsing notes

- The input is the original `finegrained-traceability-main.zip`. Its SHA-256 is `b7df2936d6304ccbe75c20f7f96f431b23861f0f9bbb90325774efade39bda62`. The package expects the original English eTour and Italian SMOS variants. [FTLR data repository](https://github.com/tobhey/finegrained-traceability)
- SMOS contains 98 Java files, one HTML/JSP artifact, and one CSS file. Its matrix names the HTML artifact `managerUser.java` for 17 links. The loader records those as links to `managerUser.html`. LiSSA uses the corresponding extensionless ID `managerUser`. All 100 targets remain in the ranking pool. [LiSSA replication package](https://github.com/ArDoCo/Replication-Package-ICSE25_LiSSA-Toward-Generic-Traceability-Link-Recovery-through-RAG)
- The automatic graph is conservative. It resolves explicit Java imports, declared inheritance, and class-qualified calls to project files. It logs external or unresolved references and leaves unsupported file types without graph edges. One eTour Java file fails to parse and remains in the ranking pool.
- Dronology remains the proposal's first public test option. The checked LiSSA revision provides the requirements and link matrix but omits the code snapshot. eANCI is an available fallback identified before test scoring. Only its file IDs and reference integrity were checked. It has 39 requirements with recorded positives, so its small effective sample should inform the power analysis. [Dronology code folder at the checked LiSSA revision](https://github.com/ArDoCo/Replication-Package-ICSE25_LiSSA-Toward-Generic-Traceability-Link-Recovery-through-RAG/tree/a8e652f29bbdcc4bdcd3d98cd85aa4a37cb65480/LiSSA-RATLR-V2/lissa/datasets/req2code/dronology-re/CC)

## Reviewing the knowledge entries

Open `knowledge/REVIEW_ME.csv` in a spreadsheet. For each row, compare the proposed glossary entry or code relationship with the cited file and excerpt. Enter `accept` or `reject`, a reviewer ID, and a short reason. For a correction, reject the original and write the suggested correction in the reason. Keep the entry IDs unchanged and return the CSV.

A glossary entry should express a project-specific equivalent or a documented abbreviation. Reject broad associations and aliases whose cited evidence does not support the proposed wording. A relationship should identify an explicit call, import, or inheritance connection between the named files. Check both endpoints. Use the source files for evidence and keep the reference link matrix and ranking outputs out of this review.

Once all entries for a project have a decision, the runner adds the verified glossary, the verified glossary plus curated relationships, and the verified glossary plus automatic graph. It uses the raw draft unchanged as the unverified comparator. A completed project batch also produces exploratory paired group bootstrap intervals for MAP and nDCG@10. The group IDs for iTrust come from its use cases. eTour and SMOS currently use one group per requirement pending a check for related scenarios.

## Rerun locally or in Colab

Use Python 3.11 or 3.12. In Colab, unzip this package into `/content`, upload or mount your dataset ZIP, then change into the `DoCIT_Phase_1` folder. Prefix these commands with `!` in a Colab code cell.

```bash
python -m pip install -r requirements.txt
python check_pipeline.py
python run_experiment.py /path/to/finegrained-traceability-main.zip --review knowledge/REVIEW_ME.csv --preview --output outputs/my_run
```

Choose a new output directory each time. Remove `--preview` to omit the extra unverified full-layer diagnostic. The package includes the exact exported development run in `outputs/preview_run`. `run_manifest.json` records input hashes, package versions, fixed weights, and graph diagnostics. Positive-link metrics exclude requirements with no recorded reference link, while the ranking export includes every requirement. The `recorded_link` column is for evaluation and should be withheld from reviewers.

## Work remaining before the held-out run

1. Review and expand the starter knowledge batch, then rerun development with the verified conditions. Audit requirement groupings before using the grouped intervals to plan the main analysis.
2. Add and benchmark the planned BGE-M3 and Qwen3 baselines on development data. Use the pilot to estimate the detectable effect, select weights on development data, and freeze the configurations, accepted knowledge, and audited matrices before scoring test projects.
3. Prepare the fresh private project. A campus equipment lending application is a candidate domain. Separate requirement writing from implementation vocabulary, preserve creation dates, and keep code and requirements private until scoring. Use a primary annotator to link requirements to implementing files with source evidence, then have a second person independently check a prespecified subset. Adjudicate before freezing the reference matrix.
4. Prepare the analyst pilot using separate tasks of comparable difficulty. Counterbalance baseline top five, verified-context top five, and deeper baseline review. Record candidate decisions and time, then compare accuracy, confidence, and review cost including preparation time.
