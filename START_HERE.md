# DoCIT results and verification

This package accompanies `DoCIT_Anonymous_Report.pdf`. It contains the Python source, executable verification notebook, exact input archive, and saved results for both experiments in the report.

| File or folder | Contents |
|---|---|
| `DoCIT_Anonymous_Report.pdf` | Anonymous report and preliminary results. |
| `DoCIT_References.bib` | BibTeX entries for the report’s sources. |
| `DoCIT_Verify_Results.ipynb` | Run the checks in Colab or Jupyter. |
| `verify_results.py` | Run the same checks from a terminal. |
| `DoCIT_Phase_1/` | Complete source for the three-project development preview, the unreviewed knowledge entries, and saved rankings and metrics. |
| `analyze_phase1.py` and `phase1_observations.csv` | Calculate and record the paired changes and context coverage reported in the draft. |
| `iTrust_pilot/` | Runnable V2.1 pilot notebook, saved results, and the script used to check average precision and reproduce the exploratory interval. |
| `data/finegrained-traceability-main.zip` | The exact input archive used in both experiments. |

## Run the verification

For Colab, extract this ZIP on your computer and open `DoCIT_Verify_Results.ipynb` in Colab. Run all cells and upload `DoCIT_Anonymous.zip` when prompted. The verification uses a CPU runtime. [Google Colab](https://colab.research.google.com/)

For a terminal, open the extracted `DoCIT_Verification` directory. Use Python 3.12 in a virtual environment and run:

```bash
python -m pip install -r requirements.txt
python verify_results.py
```

The verifier checks the input and source checksums, reruns Phase 1, compares all saved ranks and metrics, and checks the saved iTrust metrics and exploratory paired analysis. It prints the path to the fresh Phase 1 outputs. The original pilot and Phase 1 used different TF-IDF fitting rules, which the report describes separately.

## Rerun the original iTrust model inference

Open `iTrust_pilot/DoCIT_iTrust_V2_1_Pilot.ipynb` in Colab, select a CUDA GPU runtime, and provide `data/finegrained-traceability-main.zip` when prompted. For a repeat of all six configurations, set `RUN_UNIXCODER = True` and set `MODEL_REVISION` to `5604afdc964f6c53782a6813140ade5216b99006` in the configuration cell. The pilot results ZIP contains the saved rankings and metrics. Its run manifest records the original runtime versions and the resolved model commit. The verification notebook checks the archived results on CPU. Live neural inference uses the original pilot notebook. [Pinned UniXcoder checkpoint](https://huggingface.co/microsoft/unixcoder-base/tree/5604afdc964f6c53782a6813140ade5216b99006)

## Data and experimental status

The bundled dataset archive has SHA-256 `b7df2936d6304ccbe75c20f7f96f431b23861f0f9bbb90325774efade39bda62`. It comes from FTLR repository commit `197cf2f395e9e90636436f30d6383465cb9e9f63`. The archive preserves its original license and attribution files. [Pinned FTLR source and datasets](https://github.com/tobhey/finegrained-traceability/tree/197cf2f395e9e90636436f30d6383465cb9e9f63)

Phase 1 contains 15 exploratory comparisons on iTrust, eTour, and SMOS. eANCI appears in the audit only. The review decisions in `DoCIT_Phase_1/knowledge/REVIEW_ME.csv` are blank. Keep that original file for exact reproduction. A later run can use a separate file with completed decisions. [LiSSA replication package and benchmark documentation](https://doi.org/10.5281/zenodo.14714706)
