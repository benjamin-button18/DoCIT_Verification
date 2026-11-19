# DoCIT results and verification

DoCIT tests whether verified domain knowledge improves traceability between software requirements and source code.

| File or folder | Contents |
|---|---|
| `DoCIT_References.bib` | BibTeX entries for the study’s sources. |
| `DoCIT_Verify_Results.ipynb` | Run the checks in Jupyter from the repository root. |
| `verify_results.py` | Run the same checks from a terminal. |
| `DoCIT_Phase_1/` | Complete source for the three-project development preview, the unreviewed knowledge entries, and saved rankings and metrics. |
| `analyze_phase1.py` and `phase1_observations.csv` | Calculate and record the paired changes and context coverage. |
| `iTrust_pilot/` | Runnable V2.1 pilot notebook, saved results, and the script used to check average precision and reproduce the exploratory interval. |
| `data/finegrained-traceability-main.zip` | Required benchmark, available as a separate download. See [Data](#data). |

## Data

Download the benchmark separately before running any scripts or notebooks. The study uses commit `197cf2f395e9e90636436f30d6383465cb9e9f63` of the [FTLR repository](https://github.com/tobhey/finegrained-traceability).

1. Download the ZIP for that commit. [Pinned benchmark ZIP](https://github.com/tobhey/finegrained-traceability/archive/197cf2f395e9e90636436f30d6383465cb9e9f63.zip)
2. Save the download as `data/finegrained-traceability-main.zip` in this repository. Keep it zipped.

## Run the verification

From the repository root, use Python 3.12 in a virtual environment and run these commands.

```bash
python -m pip install -r requirements.txt
python verify_results.py
```

The verifier checks the benchmark contents and Phase 1 source checksums, reruns Phase 1, compares the saved ranks and metrics, and checks the saved iTrust metrics and exploratory paired analysis. It prints the path to the fresh Phase 1 outputs. The original pilot and Phase 1 used different TF-IDF fitting rules.

For Jupyter, open `DoCIT_Verify_Results.ipynb` from this directory and run all cells in a CPU runtime.

## Rerun the original iTrust model inference

Open `iTrust_pilot/DoCIT_iTrust_V2_1_Pilot.ipynb` in Colab, select a CUDA GPU runtime, and provide `data/finegrained-traceability-main.zip` when prompted. For a repeat of all six configurations, set `RUN_UNIXCODER = True` and set `MODEL_REVISION` to `5604afdc964f6c53782a6813140ade5216b99006` in the configuration cell. The pilot results ZIP contains the saved rankings and metrics. Its run manifest records the original runtime versions and the resolved model commit. The verification notebook checks the archived results on CPU. Live neural inference uses the original pilot notebook. [Pinned UniXcoder checkpoint](https://huggingface.co/microsoft/unixcoder-base/tree/5604afdc964f6c53782a6813140ade5216b99006)

## Experimental status

Phase 1 contains 15 exploratory comparisons on iTrust, eTour, and SMOS. eANCI appears in the audit only. The review decisions in `DoCIT_Phase_1/knowledge/REVIEW_ME.csv` are blank. Keep that original file for exact reproduction. A later run can use a separate file with completed decisions. [LiSSA replication package and benchmark documentation](https://doi.org/10.5281/zenodo.14714706)
