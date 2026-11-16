from pathlib import Path
import json
import nbformat

root = Path(__file__).resolve().parents[1] / 'professor-review'
path = root / 'code/main.ipynb'
n = nbformat.read(path, as_version=4)
def setcell(i, source):
    n.cells[i].source = source.strip() + '\n'

setcell(0, '''# Radiative cooling: experimental processing and Random Forest validation

This report implements the supplied project brief using the available Sample 2
instrument exports. It includes original measurements, acquisition checks, a
20-sample moving average, three-minute alignment, missing-interval exclusion,
an exported final dataset, and held-out Random Forest validation. All 20 figures
remain embedded in this notebook.

Read the [methodology](../docs/methodology.md) for the explicit adaptations from
`PLOT_TRESCO_CORRELATIONS_ML-2.m` and the [requirements audit](../docs/requirements.md).
The [final processed CSV](../processed/final_processed_dataset.csv) is the exact
physical-unit table used by the model. The five-slide presentation is outside scope.
''')
setcell(1, '''## 1. Configuration and provenance

The primary analysis uses each instrument's own timestamps (April-May 2026).
The longer April-July merged table is retained for acquisition comparison and
explicit heat-flux unit metadata, not as the primary alignment source. Later native
instrument exports are unavailable, so no independently verified full-campaign
synchronization is claimed.

The MATLAB reference applies `movmean(...,20)` before resampling. We reproduce its
centered 20-observation window (10 previous, current, 9 following) and shrinking
endpoints, within uninterrupted valid segments only. The PDF's three-minute grid,
missing-interval exclusion, five environmental predictors, and basic Random Forest
take precedence over the older MATLAB five-minute interpolation and neural network.
The window is fixed from the reference, never selected by model scores.
''')
s=n.cells[2].source.replace("import hashlib", "import hashlib\nimport json")
s=s.replace("DATA_SCOPE = 'full_campaign'  # Alternative: 'native_exports'", "DATA_SCOPE = 'native_exports'  # Primary: independently available instrument timestamps\nMOVING_WINDOW = 20\nMAX_SOURCE_GAP = pd.Timedelta('3min')")
s=s.replace("assert DATA_SCOPE in {'full_campaign', 'native_exports'}", "assert DATA_SCOPE == 'native_exports'")
setcell(2,s)
setcell(5, '''### Heat-flux units and duplicate timestamps

RAFT unit metadata are looked up at exact matching timestamps and checked against
the unchanged raw numbers. Sensitivity remains 60 microvolts per W/m². This proves
conversion consistency, not sensor calibration or polarity. The MATLAB sign reversal
belongs to a different experiment and is not imposed on these recordings.

As in MATLAB `unique(...,'stable')`, the first record at each duplicate timestamp is
retained. Sorting is stable. Duplicate and conflicting-value counts are reported.
Humidity remains percent and temperatures remain Celsius, with units explicit.
''')
s=n.cells[6].source.replace("sort_values('timestamp')", "sort_values('timestamp', kind='stable')")
s=s.replace("native_series = {name: series.groupby(level=0).mean().sort_index().rename(name)\n                 for name, series in native_series.items()}", '''duplicate_audit = pd.DataFrame([
    {'Variable': name, 'Duplicate records': int(series.index.duplicated().sum()),
     'Conflicting timestamps': int(series.groupby(level=0).nunique(dropna=False).gt(1).sum())}
    for name, series in native_series.items()]).set_index('Variable')
display(duplicate_audit)
native_series = {name: series.loc[~series.index.duplicated(keep='first')].sort_index().rename(name)
                 for name, series in native_series.items()}''')
setcell(6,s)
setcell(9, '''## 3. Missing intervals, moving average and common time grid

First audit unsmoothed values on midnight-anchored, left-closed three-minute bins.
Only bins containing finite readings of all six required variables are eligible.
Within those eligible intervals, smooth each native series **before** taking its
three-minute bin mean. This retains MATLAB's sample-based order while respecting
the PDF's prohibition on filling missing intervals.

Smoothing restarts at a missing reading, an excluded common interval, or a native
timestamp gap greater than three minutes. This explicit conservative gap rule is
an adaptation, not a parameter found in MATLAB. Short segment endpoints use fewer
than 20 readings, as reported below. No interpolation is used at any stage.

The grid defines the interval assigned to each row; the centered moving average
uses neighboring observations. Its actual source support is recorded separately.
This is offline processed-signal reconstruction, not instantaneous forecasting.
''')
setcell(10, '''def bin_series(series):
    resampler = series.resample(GRID, origin='start_day', closed='left', label='left')
    return resampler.mean(), resampler.count()

native_pairs = {name: bin_series(native_series[name]) for name in MODEL_COLS}
native_grid = pd.concat({name: pair[0] for name, pair in native_pairs.items()}, axis=1)
native_counts = pd.concat({name: pair[1] for name, pair in native_pairs.items()}, axis=1).fillna(0)
# Restrict to complete grid intervals inside the common acquisition time range.
overlap_start = max(series.index.min() for series in native_series.values())
overlap_end = min(series.index.max() for series in native_series.values())
common_index = native_grid.index[(native_grid.index >= overlap_start.ceil(GRID)) &
                                 (native_grid.index + pd.Timedelta(GRID) <= overlap_end)]
raw_aligned = native_grid.reindex(common_index)
counts = native_counts.reindex(common_index).astype(int)
complete = raw_aligned[MODEL_COLS].notna().all(axis=1)

def smooth_native(series, eligible_bins, window=MOVING_WINDOW):
    """MATLAB even-window centering, limited to uninterrupted eligible segments."""
    if window < 2 or window % 2:
        raise ValueError('Use an even sample window of at least two.')
    bins = series.index.floor(GRID)
    eligible = pd.Series(eligible_bins.reindex(bins, fill_value=False).to_numpy(), index=series.index)
    valid = series.notna() & eligible
    delta = series.index.to_series().diff()
    # A gap of more than one grid step between occupied bins crosses an excluded bin.
    bin_delta = pd.Series(bins, index=series.index).diff()
    boundaries = (~valid | ~valid.shift(fill_value=False) |
                  delta.gt(MAX_SOURCE_GAP) | bin_delta.gt(pd.Timedelta(GRID)))
    groups = boundaries.cumsum()
    result = pd.DataFrame(index=series.index, columns=['value', 'samples'], dtype=float)
    result['support_start'] = pd.NaT
    result['support_end'] = pd.NaT
    for _, segment in series.loc[valid].groupby(groups.loc[valid]):
        size = len(segment)
        position = np.arange(size)
        left = np.maximum(0, position - window // 2)
        right = np.minimum(size, position + window // 2)
        result.loc[segment.index, 'value'] = segment.rolling(window, center=True, min_periods=1).mean()
        result.loc[segment.index, 'samples'] = right - left
        result.loc[segment.index, 'support_start'] = segment.index[left].to_numpy()
        result.loc[segment.index, 'support_end'] = segment.index[right-1].to_numpy()
    assert result.loc[~valid, 'value'].isna().all()
    return result

# Independent examples verify even-window endpoint handling and gap isolation.
example_index = pd.date_range('2026-01-01', periods=40, freq='min')
example = pd.Series(np.arange(40, dtype=float), index=example_index)
example_eligible = pd.Series(True, index=example_index.floor(GRID).unique())
example_result = smooth_native(example, example_eligible)
expected = np.array([example.iloc[max(0,i-10):min(40,i+10)].mean() for i in range(40)])
np.testing.assert_allclose(example_result.value, expected)
assert example_result.samples.iloc[0] == 10 and example_result.samples.iloc[-1] == 11
gap_index = pd.to_datetime(['2026-01-01 00:00', '2026-01-01 00:01',
                           '2026-01-01 00:10', '2026-01-01 00:11'])
gap_series = pd.Series([0., 0., 100., 100.], index=gap_index)
gap_eligible = pd.Series(True, index=gap_index.floor(GRID).unique())
np.testing.assert_allclose(smooth_native(gap_series, gap_eligible).value, gap_series)
missing_example = example.copy()
missing_example.iloc[20] = np.nan
missing_result = smooth_native(missing_example, example_eligible)
assert pd.isna(missing_result.value.iloc[20])
assert missing_result.support_end.iloc[19] < example_index[20]
assert missing_result.support_start.iloc[21] > example_index[20]
excluded_example = example_eligible.copy()
excluded_example.iloc[4] = False
excluded_result = smooth_native(example, excluded_example)
assert excluded_result.loc[example_index.floor(GRID) == excluded_example.index[4], 'value'].isna().all()

smoothed_native = {name: smooth_native(native_series[name], complete) for name in MODEL_COLS}
aligned = pd.concat({name: bin_series(table.value)[0] for name, table in smoothed_native.items()}, axis=1)
aligned = aligned.reindex(common_index)[MODEL_COLS]
assert aligned.notna().all(axis=1).equals(complete)
model_data = aligned.loc[complete].copy()
rejected = aligned.index[~complete]
support_starts = pd.concat({name: table.support_start.resample(GRID).min()
                           for name, table in smoothed_native.items()}, axis=1).reindex(model_data.index)
support_ends = pd.concat({name: table.support_end.resample(GRID).max()
                         for name, table in smoothed_native.items()}, axis=1).reindex(model_data.index)
support = pd.DataFrame({'source_start': support_starts.min(axis=1),
                        'source_end': support_ends.max(axis=1)}, index=model_data.index)
assert support.notna().all().all()
assert np.isfinite(model_data.to_numpy()).all()
assert model_data.index.is_unique and model_data.index.is_monotonic_increasing
assert (counts.loc[model_data.index] >= 1).all().all()
assert len(model_data) > 100

smoothing_audit = pd.DataFrame([
    {'Variable': name, 'Smoothed readings': int(table.value.notna().sum()),
     'Shortened-window readings': int(table.samples.lt(MOVING_WINDOW).sum()),
     'Median support (min)': (table.support_end-table.support_start).dt.total_seconds().median()/60,
     'Maximum support (min)': (table.support_end-table.support_start).dt.total_seconds().max()/60}
    for name, table in smoothed_native.items()]).set_index('Variable')
display(smoothing_audit.round(3))
coverage = pd.DataFrame({'Observed bins': raw_aligned.notna().sum(),
                         'Missing bins': raw_aligned.isna().sum(),
                         'Coverage (%)': 100 * raw_aligned.notna().mean()})
display(coverage.loc[PLOT_ORDER])
display(pd.Series({'Grid slots': len(aligned), 'Complete retained bins': len(model_data),
                   'Excluded incomplete bins': len(rejected), 'Interpolated bins': 0,
                   'Moving-average samples': MOVING_WINDOW,
                   'First retained bin': model_data.index.min(),
                   'Last retained bin': model_data.index.max()}, name='Preparation').to_frame())

def missing_runs(mask):
    runs = []
    for _, segment in mask.groupby(mask.ne(mask.shift()).cumsum()):
        if segment.iloc[0]:
            runs.append({'Start': segment.index[0], 'End (exclusive)': segment.index[-1] + pd.Timedelta(GRID),
                         'Missing bins': len(segment), 'Hours': len(segment) * pd.Timedelta(GRID).total_seconds()/3600})
    return pd.DataFrame(runs, columns=['Start', 'End (exclusive)', 'Missing bins', 'Hours'])

gap_table = missing_runs(~complete).sort_values('Missing bins', ascending=False)
display(Markdown('**Largest excluded common-grid intervals**'))
display(gap_table.head(10).reset_index(drop=True))
native_gap_rows = []
for name, series in native_series.items():
    elapsed = series.index.to_series().diff()
    for timestamp in elapsed.index[elapsed.gt(MAX_SOURCE_GAP)]:
        native_gap_rows.append({'Variable': name, 'Previous reading': timestamp-elapsed.loc[timestamp],
                                'Next reading': timestamp, 'Gap minutes': elapsed.loc[timestamp].total_seconds()/60})
native_gaps = pd.DataFrame(native_gap_rows)
display(Markdown('**Largest native timestamp gaps (not interpolated)**'))
display(native_gaps.sort_values('Gap minutes', ascending=False).head(12))
display(Markdown('**Final processed table preview (physical units)**'))
display(model_data[[TARGET, 'T_amb', 'RH', 'G', 'v', 'T_sky']].head(10))
''')
s=n.cells[11].source.replace('aligned[PLOT_ORDER].notna()', 'raw_aligned[PLOT_ORDER].notna()')
s=s.replace("f'After alignment: {GRID} means; incomplete intervals excluded; no interpolation'", "f'Processed: 20-sample moving average then {GRID} means; missing intervals excluded'")
setcell(11,s)
s=n.cells[12].source.replace('three-minute grid.', 'three-minute grid after the 20-sample moving average.').replace('full-size', 'full-size')
setcell(12,s.replace('full-size figures', 'full-size figures'))
setcell(13,n.cells[13].source.replace("f'{GRID} means; incomplete intervals excluded; no interpolation'", "f'20-sample moving average then {GRID} means; gaps preserved'"))
setcell(14, '''### Compare acquisition and processing on the same time window

All six variables show the original readings, unsmoothed three-minute means and
the final moving-average-then-binned signals over the same three-day window.
Missing intervals remain gaps. This separates smoothing from bin averaging.
''')
setcell(15, '''window_start = model_data.index.min().normalize() + pd.Timedelta('2D')
window_end = window_start + pd.Timedelta('3D')
fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True, layout='constrained')
for ax, name in zip(axes, PLOT_ORDER):
    before = native_series[name].loc[window_start:window_end]
    unsmoothed = raw_aligned[name].where(complete).loc[window_start:window_end]
    after = plot_grid.loc[window_start:window_end, name]
    ax.scatter(before.index, before, color='#aaaaaa', s=5, alpha=0.6, label='Original readings')
    ax.plot(unsmoothed.index, unsmoothed, color='#2376b7', alpha=0.65, linewidth=0.7, label='Unsmoothed 3-min means')
    ax.plot(after.index, after, color='#c83d4b', linewidth=1.1, label='20-sample moving average + 3-min means')
    ax.set_ylabel(LABELS[name][1])
axes[0].legend(loc='upper right', fontsize=8)
date_axis(axes[-1])
fig.suptitle(f'Before and after processing: native timestamps, {window_start:%d %b} to {window_end:%d %b}')
plt.show()
plt.close(fig)
''')
setcell(22, '''## 5. Fixed training and validation splits

The requested shuffled 80/20 split uses seed 42. Nearby observations share centered
smoothing windows, so shuffled validation is a within-campaign reconstruction
benchmark, not an independent future forecast. No tuning is performed on it.

A second evaluation reserves the last 20% chronologically and removes at least
24 hours before it from training. Actual raw support ranges are checked to ensure
that **no training window shares source observations with validation**. This is a
stronger temporal test, although centered preprocessing still uses future neighbors
within the validation period and is unsuitable for real-time predictions.
''')
s=n.cells[23].source.replace("assert X.index[chrono_train].max() < chronological_start - CHRONOLOGICAL_GAP", "assert X.index[chrono_train].max() < chronological_start - CHRONOLOGICAL_GAP\nassert support.iloc[chrono_train].source_end.max() < support.iloc[chrono_valid].source_start.min(), 'Chronological smoothing support overlaps.'\nprint('Chronological train/validation raw support ranges are disjoint.')")
setcell(23,s)
setcell(26, '''### Measured validation values versus predictions

Following the MATLAB `linearRegressionPlot` helper, show a least-squares fitted line
and its slope/intercept alongside the 1:1 agreement line. The displayed prediction
R² compares predictions directly with measurements; fit-line R² is a separate
descriptive correlation statistic and is not a replacement for prediction accuracy.
Both panels use held-out predictions only, including negative values and extremes.
''')
s=n.cells[27].source.replace("ax.plot(bounds, bounds, '--', color='black', linewidth=1, label='1:1 agreement')", """ax.plot(bounds, bounds, '--', color='black', linewidth=1, label='1:1 agreement')
    slope, intercept = np.polyfit(result['Measured'], result['Predicted'], 1)
    line_x = np.asarray(bounds)
    ax.plot(line_x, slope*line_x+intercept, color='#2376b7', linewidth=1.2,
            label=f'Fitted: y = {slope:.3f}x {intercept:+.2f}')
    fit_r2 = r2_score(result['Predicted'], slope*result['Measured']+intercept)""")
s=s.replace("f\"MAE = {values['MAE (W/m²)']:.2f} W/m²\\nR² = {values['R²']:.3f}\")", "f\"MAE = {values['MAE (W/m²)']:.2f} W/m²\\nPrediction R² = {values['R²']:.3f}\\nFit-line R² = {fit_r2:.3f}\")")
setcell(27,s)
s=n.cells[31].source.replace("observed {GRID} bins", "processed {GRID} bins")
s=s.replace("summary_lines.append('- Verify", "summary_lines.append('- The target is the 20-sample-smoothed, three-minute mean recorded flux. Random validation shares smoothing support; chronological support is disjoint.')\nsummary_lines.append('- Verify")
setcell(31,s)
setcell(32, '''## 9. Export the final dataset and verify traceability

`processed/final_processed_dataset.csv` contains exactly the physical-unit model
table, with timestamp, Q_net, T_amb, RH, G, v and T_sky. It opens in Excel and is
read back below to verify numerical equivalence. Missing intervals are absent,
so retained timestamps can be more than three minutes apart.

`results/bin_audit.csv` records eligibility, native counts, unsmoothed values,
per-variable smoothing support and the common interval end. Companion files record
split membership, held-out predictions, metrics, native gaps and processing metadata.
These supply the chain from input hashes to final data and validation.
''')
s=n.cells[33].source
s=s[:s.index("print('Checks passed")]+'''
output_columns = [TARGET, 'T_amb', 'RH', 'G', 'v', 'T_sky']
export_table = model_data[output_columns].rename_axis('timestamp')
results_dir = PROJECT_ROOT / 'results'
results_dir.mkdir(exist_ok=True)
final_path = DATA_DIR / 'final_processed_dataset.csv'
export_table.to_csv(final_path, float_format='%.12g', date_format='%Y-%m-%d %H:%M:%S')
roundtrip = pd.read_csv(final_path, parse_dates=['timestamp']).set_index('timestamp')
pd.testing.assert_index_equal(roundtrip.index, export_table.index)
np.testing.assert_allclose(roundtrip, export_table, rtol=1e-10, atol=1e-10)

audit = pd.DataFrame({'interval_end': aligned.index + pd.Timedelta(GRID),
                      'retained': complete}, index=aligned.index)
audit.index.name = 'timestamp'
for name in MODEL_COLS:
    audit[name + '_native_count'] = counts[name]
    audit[name + '_unsmoothed_mean'] = raw_aligned[name]
    audit[name + '_support_start'] = support_starts[name]
    audit[name + '_support_end'] = support_ends[name]
audit['missing_variables'] = raw_aligned.isna().apply(lambda row: ','.join(row.index[row]), axis=1)
audit.to_csv(results_dir / 'bin_audit.csv', float_format='%.12g')
split_export = support.copy()
for name, (train, valid) in splits.items():
    split_export[name] = 'excluded_gap'
    split_export.loc[X.index[train], name] = 'train'
    split_export.loc[X.index[valid], name] = 'validation'
split_export.to_csv(results_dir / 'split_assignments.csv', index_label='timestamp')
prediction_export = pd.concat(predictions, names=['evaluation', 'timestamp'])
prediction_export.to_csv(results_dir / 'validation_predictions.csv', float_format='%.12g')
metrics.to_csv(results_dir / 'metrics.csv', index=False, float_format='%.12g')
native_gaps.to_csv(results_dir / 'native_gaps.csv', index=False)
reference = PROJECT_ROOT / 'references/PLOT_TRESCO_CORRELATIONS_ML-2.m'
metadata = {
    'scope': DATA_SCOPE, 'grid': GRID, 'moving_average_samples': MOVING_WINDOW,
    'moving_average': 'Centered: 10 previous, current, 9 following; shrink within uninterrupted valid segments',
    'max_native_gap': str(MAX_SOURCE_GAP), 'interpolation': False,
    'source_units': {'Q_net':'W/m^2 (recorded sign)', 'T_amb':'deg C', 'RH':'%', 'G':'W/m^2', 'v':'m/s', 'T_sky':'deg C'},
    'rows': len(model_data), 'grid_slots': len(aligned), 'excluded_bins': len(rejected),
    'first_timestamp': str(model_data.index.min()), 'last_timestamp': str(model_data.index.max()),
    'common_acquisition_start': str(overlap_start), 'common_acquisition_end': str(overlap_end),
    'seed': SEED, 'validation_fraction': VALIDATION_FRACTION,
    'chronological_gap': str(CHRONOLOGICAL_GAP),
    'chronological_source_support_disjoint': bool(support.iloc[chrono_train].source_end.max() < support.iloc[chrono_valid].source_start.min()),
    'random_split_support_overlap': 'Expected; shuffled score is reconstruction only',
    'source_hashes': source_inventory.SHA256.to_dict(),
    'matlab_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
    'versions': versions.to_dict(),
    'artifacts': {}
}
for artifact in [final_path, *sorted(results_dir.glob('*.csv'))]:
    metadata['artifacts'][artifact.relative_to(PROJECT_ROOT).as_posix()] = hashlib.sha256(artifact.read_bytes()).hexdigest()
(results_dir / 'processing_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Passed: unchanged inputs, MATLAB window examples, gap isolation, complete bins,')
print('disjoint chronological source support, training-only scaling, held-out metrics and CSV round trip.')
print(f'Exported {len(export_table):,} rows to processed/final_processed_dataset.csv.')
print('All 20 figures are embedded in main.ipynb; no separate figure files were written.')
'''
setcell(33,s)
for cell in n.cells:
    if cell.cell_type == 'code':
        cell.outputs=[]
        cell.execution_count=None
nbformat.write(n,path)
for file in [root/'scripts/run_notebook.py',root/'scripts/validate_submission.py']:
    file.write_text(file.read_text(encoding='utf-8').replace('main_5.ipynb','main.ipynb'), encoding='utf-8')
reference_dir=root/'references'
reference_dir.mkdir(exist_ok=True)
source=root.parent/'PLOT_TRESCO_CORRELATIONS_ML-2.m'
(reference_dir/source.name).write_bytes(source.read_bytes())
print('Updated notebook and filename references; copied unchanged MATLAB reference.')
