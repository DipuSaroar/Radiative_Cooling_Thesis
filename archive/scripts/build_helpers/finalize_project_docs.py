from pathlib import Path
import json
import hashlib
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'professor-review'
metadata = json.loads((root/'results/processing_metadata.json').read_text())
metrics = pd.read_csv(root/'results/metrics.csv')
scores = metrics.loc[metrics.Model.eq('Random Forest') & metrics.Partition.eq('Validation')]
score_rows='\n'.join(f"| {r['Evaluation']} | {int(r['N']):,} | {r['RMSE (W/m²)']:.2f} | {r['MAE (W/m²)']:.2f} | {r['R²']:.3f} |" for _,r in scores.iterrows())
readme = '''# Predicting net heat flux in passive radiative cooling materials

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
SCORE_ROWS

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
| [code/main.ipynb](code/main.ipynb) | Complete executed workflow and all figures |
| [processed/final_processed_dataset.csv](processed/final_processed_dataset.csv) | Final Excel-readable model dataset |
| [processed/README.md](processed/README.md) | Input and output columns, units and provenance |
| [results/](results/) | Bin audit, split assignments, predictions, metrics, gaps and hashes |
| [docs/methodology.md](docs/methodology.md) | Processing choices, MATLAB adaptations and limitations |
| [docs/requirements.md](docs/requirements.md) | Requirement-by-requirement audit against the PDF |
| [references/](references/) | Unchanged MATLAB reference and source description |
| data_manifest.json | Original input file sizes, row counts, periods and hashes |

The requested five-minute presentation is intentionally outside this submission's
current scope. Earlier exploratory work remains in development branches. Private
audio is not redistributed.

## Reproduce

Use Python 3.11 and the pinned dependencies in `requirements.txt`.

Windows PowerShell:

```powershell
git clone --branch professor-review --single-branch https://github.com/al-rafi-DSC/Dipu_Bhai_Thesis.git
cd Dipu_Bhai_Thesis
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
'''.replace('SCORE_ROWS',score_rows)
(root/'README.md').write_text(readme,encoding='utf-8')

methodology='''# Methodology and interpretation

## Objective and reference hierarchy

Predict processed recorded net heat flux from ambient temperature, relative humidity,
solar irradiance, wind speed and sky temperature. The project brief (`summary.pdf`)
specifies a basic Random Forest, a common three-minute grid, removal of incomplete
intervals, original/processed plots, a final Excel-readable dataset and held-out
validation. The supplied MATLAB file specifies the previously missing moving-average
window. Its older neural-network workflow is not the current model requirement.

## Acquisition and scope

The four input files remain byte-for-byte unchanged, verified by `data_manifest.json`.
The primary model uses the available April-May instrument exports and their own
timestamps, restricted to full three-minute intervals inside their shared acquisition
period. The April-July merge is used for acquisition comparison and exact-timestamp
RAFT unit lookup, not to supply model alignment. This narrows the modelling period
because the brief makes auditable synchronization the priority.

The RAFT `HEAT_FLUX_V` field contains both volts and microvolts. Exact-timestamp
metadata from the existing merged table identify units, and matching raw readings
are asserted. Voltages are divided by the inherited sensitivity 60e-6 V/(W/m²).
Recorded polarity is retained; the sign reversal in the 2025 MATLAB experiment is
not evidence for reversing the 2026 dataset. The files are consolidated exports,
not independently verified original logger archives.

IR uses `TPROC` for detailed records and `TEMPERATURE_C` for simple records.
Stable sorting followed by first-record duplicate removal mirrors MATLAB's
`unique(...,'stable')`. Duplicate conflicts are reported, not silently averaged.
Temperatures remain Celsius and RH remains percent. These units are labelled in
the data dictionary and are consistent with subsequent training-only normalization.
Sample temperature is not a predictor or an additional completeness requirement.

## Synchronization and smoothing

1. Parse and inspect original instrument timestamps before modifying signals.
2. Audit raw finite values in midnight-anchored, left-labelled, left-closed three-minute
   bins. A row is eligible only when all six required variables have readings.
   Partial bins at the common acquisition boundaries are excluded.
3. Apply a 20-sample centered moving mean separately to each native series within
   eligible, uninterrupted segments. An even window includes ten preceding readings,
   the current reading and nine following readings. Endpoints shrink. This matches
   MATLAB's sample-window convention; it does not mean 20 minutes or 20 resampled bins.
4. Reset the smoother at non-finite readings, excluded common intervals, or native
   gaps greater than three minutes. This conservative, explicitly declared rule
   adapts the older script to the brief's missing-data requirement. No smoothing
   support crosses an excluded interval. Shortened windows and actual durations
   are reported, not hidden.
5. Average the smoothed native readings into the same three-minute bins. Remove
   incomplete bins and retain the original eligibility mask. Never interpolate.

The grid specifies which interval a row represents. Centered smoothing uses nearby
intervals too, so per-variable source-support bounds are exported in `bin_audit.csv`.
Within-bin instrument timing differences and undocumented clock offsets can remain.
No timezone or clock correction is guessed. Missing positions are restored as NaN
only for plotting so lines do not bridge outages.

MATLAB moving-mean semantics are documented by
[MathWorks](https://www.mathworks.com/help/matlab/ref/movmean.html).

## Explicit differences from the MATLAB reference

| Reference code | Current implementation and reason |
|---|---|
| 2025 Excel input and custom `sumtime` | Actual 2026 timestamps from the supplied CSVs. The helper's calendar/offset logic is not transplanted. |
| `movsetting = 20` before resampling | Preserved on native samples, with gap resets to satisfy the brief. |
| Five-minute linear interpolation | Three-minute bin means; no interpolation, per the newer brief. |
| `Q_orig = -Data1_p(:,2)` | Retain recorded sign pending polarity evidence for these sensors. |
| Kelvin temperatures, RH fraction | Celsius and percent, clearly labelled; scaling fitted on training data. |
| `x1 = T_s_full` | Ambient temperature, as requested by the brief. |
| Global normalization and NN tuning | Training-only scaling and untuned Random Forest. |
| Final predictions on training data | Report scores from held-out observations only. |
| `linearRegressionPlot` scatter and fitted line | Same comparison components plus a 1:1 line and direct prediction metrics. |
| Undefined simulation variables | Excluded: incomplete simulation code is unrelated to the requested baseline. |

## Evaluation

The model uses 100 trees, unrestricted depth, one sample per leaf, all five features
available per split, bootstrap sampling and seed 42. No hyperparameters or smoothing
settings are chosen from validation results. A min-max scaler lives inside each
training pipeline. The target remains in physical units.

The requested shuffled 80/20 split is a within-campaign reconstruction benchmark.
Rows are disjoint, but centered smoothing and serial dependence mean nearby rows
share raw support. Its score must not be described as independent future performance.

The chronological evaluation reserves the final 20% and removes the 24 hours before
the boundary from training. Actual source-support ranges are checked for complete
train/validation separation. It evaluates a later processed period, not a causal
real-time forecast: centered validation inputs themselves use future neighbors.
Only one later period is tested. A final all-data fit has no independent test score.

Both splits include MAE, RMSE, direct prediction R², prediction bias and a training-mean
baseline. Training scores diagnose overfitting. The comparison figure adds a fitted
line following MATLAB's helper. Its fit-line R² describes association between
measurements and predictions and is distinct from direct prediction R².
Permutation importance describes model reliance, not physical causality.

## Exports and reproducibility

`processed/final_processed_dataset.csv` is exactly the physical-unit table used to
build X and y, read back and numerically verified. All rows lie on the common grid;
excluded intervals make some successive timestamps farther than three minutes apart.
`results/` contains bin eligibility/counts/source support, native gaps, train/validation
membership, held-out predictions, metrics, source hashes, parameters and output hashes.
The validator independently recomputes held-out metrics and checks support separation.
The notebook tests window centering/endpoints, missing readings and gap isolation.

## Limits

Calibration and polarity, instrument clock synchronization and the apparent -50 °C
IR floor require measurement-level confirmation. No target clipping or floor
replacement is applied. Raw export consolidation itself is inherited. The later
May-July campaign cannot be independently synchronized without later native files.
The results are evidence about this processed dataset, not certified physical
cooling performance or real-time deployment accuracy.

## References

- Supplied `summary.pdf`, 11 pages, reviewed 29 September 2026; presentation excluded.
- Supplied `PLOT_TRESCO_CORRELATIONS_ML-2.m`, retained unchanged under `references/`.
- Bertiglia et al., *An Ambient Temperature-tracking Setup for Outdoor Characterization
  of Passive Radiative Cooling (PRC) Materials*, UIT 2026 presentation, 23 June 2026.
- Supervisor's approximately six-minute recording supplied with this project,
  filename dated 18 September 2026. The recording is not redistributed.
'''
(root/'docs/methodology.md').write_text(methodology,encoding='utf-8')
p=root/'processed/README.md'
s=p.read_text(encoding='utf-8').replace('# Input data','# Experimental input and processed data').replace('code/main_5.ipynb','code/main.ipynb')
s=s.replace('Default full-campaign input and explicit RAFT unit metadata','Acquisition comparison and explicit RAFT unit metadata')
s += '''
## Final processed dataset

`final_processed_dataset.csv` is generated by `code/main.ipynb` and is the exact table
used by the model. It contains 11,304 rows, 20 April to 15 May 2026, after the native
20-sample moving average and three-minute bin means. These are processed values,
not individual raw measurements. Column order and units:

| Column | Meaning | Unit |
|---|---|---|
| timestamp | Start of left-closed three-minute interval | Recorded local clock; timezone unknown |
| Q_net | Processed recorded-sign net heat flux, target | W/m² |
| T_amb | Processed ambient temperature | °C |
| RH | Processed relative humidity | % |
| G | Processed solar irradiance | W/m² |
| v | Processed wind speed | m/s |
| T_sky | Processed apparent sky temperature | °C |

Open the CSV directly in Excel, or use Data > From Text/CSV with comma as delimiter.
No missing values are filled: absent intervals do not appear as rows. Every retained
timestamp is on the three-minute grid, but the retained table is not continuous.
`../results/bin_audit.csv` records all common grid slots, their eligibility, counts,
unsmoothed bin means and per-variable source-support bounds. Counts mean finite
native readings inside each bin, not the number of samples in the moving window.
The source-support bounds describe neighboring readings used by that window.
'''
p.write_text(s,encoding='utf-8')
(root/'results/README.md').write_text('''# Reproducible processing and validation outputs

Generated by `code/main.ipynb`. Final model values are in
`../processed/final_processed_dataset.csv` with units in its data dictionary.

| File | Row meaning and purpose |
|---|---|
| bin_audit.csv | One common three-minute slot. Eligibility, interval end, per-variable finite native counts, unsmoothed means, smoothing support start/end, and missing variable names. Excluded rows have no smoothing support. |
| native_gaps.csv | One native timestamp gap greater than three minutes for one required variable. Variables sharing an instrument can have the same gap. This threshold is a declared processing rule. |
| split_assignments.csv | One retained model row. Both split labels and the earliest/latest raw support across all variables. `excluded_gap` rows participate in the final full-data fit but not that evaluation's training or validation. |
| validation_predictions.csv | One held-out row per evaluation. Measured means processed target; predictions and residuals are W/m². The same timestamp may occur in both evaluations. |
| metrics.csv | One model/evaluation/partition. N, RMSE and MAE (W/m²), direct prediction R², and prediction-minus-measurement bias (W/m²). |
| processing_metadata.json | Scope, grid, sample window, gap rule, units, row counts, versions, source/reference hashes and exported-file hashes. |

The random split shares neighboring smoothing support. Chronological train/validation
source ranges are disjoint. Neither evaluation is a real-time forecasting experiment.
`scripts/validate_submission.py` checks export hashes and independently recalculates
validation metrics. Re-execution regenerates these files; no separate figures exist.
''',encoding='utf-8')
pdf=Path(r'C:\Users\Sayed Al Rafi\Downloads\summary.pdf')
(root/'references/README.md').write_text('''# Methodology references

`PLOT_TRESCO_CORRELATIONS_ML-2.m` is the unchanged MATLAB file supplied for this
project. It supplies `movsetting = 20`, native-sample smoothing before resampling,
and the `linearRegressionPlot` scatter/fitted-line helper. It is a reference, not an
executable dependency: its older Excel data, neural-network optimization and
unfinished simulation code are not required to reproduce the Python analysis.

The supplied 11-page `summary.pdf` defines the current requirements. Its content is
mapped in `../docs/requirements.md`; the private document is not needed to run the
notebook. The five-minute presentation is excluded from this task.

Original source fingerprints:

- MATLAB SHA256: `MATLAB_HASH`
- Summary PDF SHA256: `PDF_HASH`

See `../docs/methodology.md` for every material adaptation from the MATLAB reference.
'''.replace('MATLAB_HASH',hashlib.sha256((root/'references/PLOT_TRESCO_CORRELATIONS_ML-2.m').read_bytes()).hexdigest()).replace('PDF_HASH',hashlib.sha256(pdf.read_bytes()).hexdigest()),encoding='utf-8')
print('Updated README, methodology, dictionaries and source references.')
