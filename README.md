# Predicting net heat flux in passive radiative cooling materials

Experimental-data processing and an untuned Random Forest for the Sample 2 RAFT
campaign at INRiM, Turin.

**Open [the executed notebook, `code/main.ipynb`](code/main.ipynb).** All 20 figures
are embedded. The [final processed dataset](processed/final_processed_dataset.csv)
opens directly in Excel and contains the exact physical-unit table used by the model.

## What is implemented

The analysis inventories the available original-timestamp instrument exports, checks
units and duplicates, plots acquisition data, excludes incomplete intervals and
applies the MATLAB reference's **20-sample centered moving average before three-minute
binning**. Smoothing resets at missing/excluded intervals and native gaps over three
minutes. There is no interpolation or target clipping.

The final dataset contains **11,304 observations**, from **20 April 2026 09:45** to
**15 May 2026 11:57**. Within the shared acquisition range, 743 of 12,047 complete
three-minute grid slots are excluded for missing variables. Partial boundary bins
are also excluded. The model uses five atmospheric inputs, training-only min-max
scaling, a shuffled 80/20 split with seed 42, and a chronological holdout.

## Executed results

| Evaluation | Validation rows | RMSE (W/m²) | MAE (W/m²) | Prediction R² |
|---|---:|---:|---:|---:|
| Random 80/20 | 2,261 | 7.91 | 3.21 | 0.944 |
| Chronological holdout | 2,261 | 18.64 | 11.34 | 0.604 |

The random split shares neighboring smoothing support and describes reconstruction
within this campaign. The chronological split has a 24-hour training purge and
verified disjoint source support. Its performance concerns a later processed period;
centered smoothing still makes this an offline analysis, not a real-time forecast.
The target is the smoothed, binned recorded heat flux. These scores are not directly
comparable with the earlier unsmoothed April-July run because both processing and
scope changed. No model or window selection was based on these holdout results.

## Files for review

| File | Purpose |
|---|---|
| [code/data_cleaning.ipynb](code/data_cleaning.ipynb) | Step 1: rebuilds the three merged instrument CSVs from `raw_data/` and verifies they are byte-identical |
| [code/main.ipynb](code/main.ipynb) | Step 2: complete executed workflow and all figures |
| [processed/final_processed_dataset.csv](processed/final_processed_dataset.csv) | Final Excel-readable model dataset |
| [processed/README.md](processed/README.md) | Input and output columns, units and provenance |
| [results/](results/) | Bin audit, split assignments, predictions, metrics, gaps and hashes |
| [docs/methodology.md](docs/methodology.md) | Processing choices, MATLAB adaptations and limitations |
| [docs/requirements.md](docs/requirements.md) | Requirement-by-requirement audit against the PDF |
| [references/](references/) | Unchanged MATLAB reference, source description and the RAFT presentation PDF |
| [raw_data/](raw_data/) | Original instrument log files (RAFT, meteo, IR) as first uploaded |
| [archive/](archive/) | Earlier exploratory notebooks, datasets, figures and scripts (not used by the final analysis) |
| data_manifest.json | Original input file sizes, row counts, periods and hashes |

The requested five-minute presentation is intentionally outside this submission's
current scope. Earlier exploratory work is kept in `archive/` and in the development branches. Private
audio is not redistributed.

## Web app (runs locally)

An interactive app for presenting the project lives in `app/`. Double-click **`run_app.bat`**
(or run `.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py`), then open
http://localhost:8501. The first run creates `.venv` and installs `requirements.txt` and
`app/requirements.txt`.

| Page | What it shows |
|---|---|
| Overview | Project summary, pipeline diagram, headline results |
| Data and graphs | Gap-free time series of all six variables, smoothing before/after, daily cycle, correlations, clean CSV download |
| Model performance | Random Forest on the random 80/20 split and the chronological holdout, feature importance, errors, live prediction tool |
| Upload new data | Upload raw RAFT/weather/IR files or a CSV; they are cleaned with the thesis pipeline, analysed on their own or added to the campaign |

`app/pipeline.py` reuses the notebook logic and reproduces `processed/final_processed_dataset.csv`
and `results/metrics.csv` exactly from `raw_data/`. Uploads stay in the browser session; no project
file is changed.

## Reproduce

Use Python 3.11 and the pinned dependencies in `requirements.txt`.

Windows PowerShell:

```powershell
git clone https://github.com/DipuSaroar/Radiative_Cooling.git "Radiative Cooling"
cd "Radiative Cooling"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/validate_submission.py
.\.venv\Scripts\python.exe scripts/run_notebook.py
```

macOS/Linux, after cloning:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/validate_submission.py
.venv/bin/python scripts/run_notebook.py
```

The runner uses the selected interpreter, executes every cell, checks source integrity
and exported results, then saves the executed notebook. It regenerates the final CSV
and `results/` files. No standalone figures are created. The optional
`--output outputs/main_reexecuted.ipynb` changes only the notebook destination; data
exports still go to their documented repository paths.

For interactive use, open `code/main.ipynb` and choose the same Python environment.
The saved report is readable without running it. Validation checks all input/output
hashes, complete model rows, chronological support separation, saved execution and
independently recomputed held-out metrics.

## Source limits

The primary analysis now uses the available native-export overlap so synchronization
can be inspected directly. Later April-July native files are unavailable; the longer
merge is retained for inspection and explicit RAFT unit metadata, not primary model
alignment. Original export consolidation, sensor calibration/polarity, clock offsets,
and the apparent -50 °C sky-temperature floor remain measurement-level limitations.
Software checks establish reproducibility, not independent certification of sensors.
