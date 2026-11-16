from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'Dipu_Bhai_Thesis/code/main_5.ipynb'
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md('''
# Radiative cooling: acquisition checks to Random Forest validation

**Complete baseline workflow · main_5**

Predict recorded net heat flux, $Q_{net}$ (W/m²), from ambient temperature, relative
humidity, wind speed, solar irradiance and sky temperature. All figures, tables and
evaluation results are embedded below. Run **Restart Kernel and Run All** to reproduce them.
This notebook does not write separate figures, change input files, or depend on
executing `main_clean.ipynb` or `main_3.ipynb`.

The latest supervisor audio takes precedence over the older presentation's MLP example:

| Stage | Supervisor's request | Implementation here |
|---|---|---|
| Acquisition, 00:25–01:40 | Inspect each instrument on its own timestamps before preprocessing | Source inventory, conversion audit and original-timestamp plots |
| Preparation, 01:55–02:45 | Align variables; remove intervals with missing variables; show the plots again | Three-minute bin means, observed bins only, explicit missing intervals |
| Modelling, 02:45–03:20 | Normalize, split, use a basic Random Forest | Training-only min–max scaler and an untuned regressor |
| Evaluation, 03:20–03:50 | Metrics and measured validation values versus predictions | RMSE, MAE, R², mean baseline and parity plots |
| Scope, 04:00–05:40 | Keep the Python workflow simple; no network optimization needed | No hyperparameter search, no neural network |

Sources: [supervisor audio](../prof/WhatsApp%20Audio%202026-09-18%20at%2012.29.03.mp4)
(timestamps refer to an automated transcript), and
[RAFT presentation](../prof/RAFT_presentation-1.pdf), particularly slides 9, 11–13.
The audio mentions a **three-minute** step; the older slide shows five minutes.
`GRID` makes this choice explicit. The referenced MATLAB code is not present in this
repository, so this implements the described framework, not a line-by-line translation.
''')

md('''
## 1. Configuration and provenance

The separate instrument exports end on **15 May 2026**. The existing, unsmoothed
`merged_dataset_104957.csv` extends into July. The default `full_campaign` mode uses
that full table without using `model_input_5min.csv`. Original-timestamp inspection and
an independently rebuilt overlap comparison use the separate instrument exports.

The full table has already had weather/sky readings assigned to RAFT timestamps by
an earlier merge. Its original matching tolerance and later native timestamps are not
available here. Consequently, full-campaign results are conditional on that earlier
merge; they are not proof that its acquisition/alignment is correct. Set
`DATA_SCOPE = "native_exports"` to run the entire model using only the available native
instrument timestamps, with fresh alignment in this notebook. No data from the two
paths are silently combined.

Heat-flux calibration is inherited from the existing dataset: `Q_net = voltage_V / 60e-6`.
This coefficient can be checked for arithmetic consistency, but no calibration
certificate is supplied. Preserve the recorded sign: the slides use positive cooling
power, whereas this export is often negative at night. Sensor polarity must be verified
before assigning a physical heating/cooling interpretation; no sign flip is guessed.

Dependencies: Python 3.11+, `numpy`, `pandas`, `matplotlib`, `scikit-learn`,
`IPython`, and a Jupyter kernel. Package versions and source hashes are recorded below.
''')

code('''
%matplotlib inline
from pathlib import Path
import hashlib
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn
from IPython.display import display, Markdown
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

def find_project_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for candidate in (base, base / 'Dipu_Bhai_Thesis'):
            if (candidate / 'processed/merged_dataset_104957.csv').is_file():
                return candidate
    raise FileNotFoundError('Open this notebook from the project or its code folder.')

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / 'processed'
DATA_SCOPE = 'full_campaign'  # Alternative: 'native_exports'
GRID = '3min'
SEED = 42
VALIDATION_FRACTION = 0.20
CHRONOLOGICAL_GAP = pd.Timedelta('1D')
HEAT_FLUX_SENSITIVITY_V = 60e-6  # V per (W/m²), inherited, not independently certified
FEATURES = ['T_amb', 'RH', 'v', 'G', 'T_sky']
TARGET = 'Q_net'
MODEL_COLS = FEATURES + [TARGET]
PLOT_ORDER = [TARGET, 'G', 'T_amb', 'T_sky', 'RH', 'v']
LABELS = {
    'Q_net': ('Recorded net heat flux', 'Q_net [W/m²]', '#202020'),
    'G': ('Solar irradiance', 'G [W/m²]', '#dc8b12'),
    'T_amb': ('Ambient temperature', 'T_amb [°C]', '#2376b7'),
    'T_sky': ('Sky temperature', 'T_sky [°C]', '#c83d4b'),
    'RH': ('Relative humidity', 'RH [%]', '#329261'),
    'v': ('Wind speed', 'v [m/s]', '#8661b3'),
}
assert DATA_SCOPE in {'full_campaign', 'native_exports'}
plt.rcParams.update({'figure.dpi': 105, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.2})
pd.set_option('display.max_columns', 12)
pd.set_option('display.precision', 4)
versions = pd.Series({'Python': platform.python_version(), 'pandas': pd.__version__,
                      'NumPy': np.__version__, 'Matplotlib': matplotlib.__version__,
                      'scikit-learn': sklearn.__version__}, name='Version')
display(versions.to_frame())
print(f'Scope: {DATA_SCOPE}; grid: {GRID}; random seed: {SEED}')
''')

md('''
## 2. Read and inspect the available acquisitions

These are per-instrument CSV exports, not the original `.dat`/logger files. Read all
records before any modelling exclusions. Numeric parsing failures and non-finite values
become missing values and are counted. Timestamp errors stop execution. Relative
humidity stays in percent. `TEMPERATURE_C` supplies the simple sky export and `TPROC`
supplies the detailed sky export; camera-internal temperatures are not predictors.
''')
code('''
source_names = {
    'RAFT export': 'merged_raft.csv',
    'Weather export': 'merged_meteo.csv',
    'Sky export': 'merged_ir.csv',
    'Full merged table': 'merged_dataset_104957.csv',
}
source_frames = {name: pd.read_csv(DATA_DIR / filename, low_memory=False)
                 for name, filename in source_names.items()}
raft = source_frames['RAFT export'].copy()
weather = source_frames['Weather export'].copy()
sky = source_frames['Sky export'].copy()
full = source_frames['Full merged table'].copy()
raft['timestamp'] = pd.to_datetime(raft['DATE'].str.strip() + ' ' + raft['TIME'].str.strip(),
                                  format='%d/%m/%Y %H:%M:%S', errors='raise')
# Remove the English weekday and map the English month explicitly (locale-independent).
month_numbers = dict(zip(['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December'],
                        [f'{m:02}' for m in range(1, 13)]))
weather_time = weather['TIME'].str.extract(
    r'^\\w+ (\\d{1,2}) (\\w+) (\\d{4}) (\\d{2}:\\d{2}:\\d{2})$')
weather['timestamp'] = pd.to_datetime(
    weather_time[2] + '-' + weather_time[1].map(month_numbers) + '-' +
    weather_time[0].str.zfill(2) + ' ' + weather_time[3], errors='raise')
sky['timestamp'] = pd.to_datetime(sky['TIMESTAMP'], format='mixed', errors='raise')
full['timestamp'] = pd.to_datetime(full['timestamp'], errors='raise')
for name, frame in [('RAFT', raft), ('Weather', weather), ('Sky', sky), ('Full', full)]:
    assert frame['timestamp'].notna().all(), f'{name}: missing/invalid timestamps'
assert not full['timestamp'].duplicated().any(), 'Full table has ambiguous duplicate timestamps.'

def numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)

for column in MODEL_COLS + ['Q_net_raw', 'Q_net_voltage_V', 'T_pt100_C']:
    full[column] = numeric(full[column])
full = full.sort_values('timestamp').set_index('timestamp')
inventory = []
for name, frame in [('RAFT export', raft), ('Weather export', weather),
                    ('Sky export', sky), ('Full merged table', full.reset_index())]:
    times = frame['timestamp'].sort_values()
    steps = times.diff().dropna()
    inventory.append({'Source': name, 'Rows': len(frame), 'First': times.min(),
                      'Last': times.max(), 'Duplicate timestamps': int(times.duplicated().sum()),
                      'Median positive step (s)': steps[steps > pd.Timedelta(0)].dt.total_seconds().median(),
                      'SHA256': hashlib.sha256((DATA_DIR / source_names[name]).read_bytes()).hexdigest()})
source_inventory = pd.DataFrame(inventory).set_index('Source')
display(source_inventory.drop(columns='SHA256'))
display(source_inventory[['SHA256']])
display(sky.groupby('RECORD_TYPE').agg(rows=('TIMESTAMP', 'size'),
                                      first=('timestamp', 'min'), last=('timestamp', 'max')))
''')

md('''
### Heat-flux units and duplicate timestamps

The RAFT export's `HEAT_FLUX_V` header is misleading: values switch between volts and
microvolts. Use the existing full table's explicit `Q_raw_unit` at the **exact same
RAFT timestamp**, verify that the raw numerical values match, then convert to volts.
No nearest-time lookup or magnitude-based unit guessing is used. One missing reading
must remain missing, not become zero. This unit metadata dependency also applies in
`native_exports` mode.

For duplicate timestamps within an instrument, average the finite readings at that
exact timestamp. Count the duplicates in the inventory above. Do not discard a whole
row simply because a different sensor or an IR diagnostic is missing.
''')
code('''
raft = raft.sort_values('timestamp').set_index('timestamp')
unit_lookup = full[['Q_raw_unit', 'Q_net_raw']].reindex(raft.index)
raft_raw = numeric(raft['HEAT_FLUX_V'])
assert np.allclose(raft_raw, unit_lookup['Q_net_raw'], equal_nan=True), (
    'RAFT raw values disagree with the exact-timestamp unit metadata source.')
unit_scale = unit_lookup['Q_raw_unit'].map({'V': 1.0, 'uV': 1e-6})
assert unit_scale[raft_raw.notna()].notna().all(), 'Unknown unit for a finite RAFT value.'
raft_voltage = raft_raw * unit_scale
raft_flux = raft_voltage / HEAT_FLUX_SENSITIVITY_V
known = full['Q_raw_unit'].map({'V': 1.0, 'uV': 1e-6})
assert known[full['Q_net_raw'].notna()].notna().all()
assert np.allclose(full['Q_net_raw'] * known, full['Q_net_voltage_V'], equal_nan=True)
assert np.allclose(full['Q_net_voltage_V'] / HEAT_FLUX_SENSITIVITY_V,
                   full[TARGET], equal_nan=True)
assert np.allclose(raft_flux, full[TARGET].reindex(raft.index), equal_nan=True)
display(pd.DataFrame({'Rows': full.groupby('Q_raw_unit', dropna=False).size()}))
print('Raw-to-voltage and voltage-to-flux arithmetic agree with the existing full table.')
print('The sign and calibration coefficient remain inherited measurement assumptions.')

weather = weather.sort_values('timestamp').set_index('timestamp')
sky = sky.sort_values('timestamp').set_index('timestamp')
assert sky['RECORD_TYPE'].isin(['SIMPLE_TXT', 'DETAILED_DAT']).all()
sky_temperature = numeric(sky['TEMPERATURE_C']).where(
    sky['RECORD_TYPE'].eq('SIMPLE_TXT'), numeric(sky['TPROC']))
native_series = {TARGET: raft_flux, 'T_sky': sky_temperature}
for original, name in {'TEMP': 'T_amb', 'RH': 'RH', 'SPEED': 'v', 'SOLARRAD': 'G'}.items():
    native_series[name] = numeric(weather[original])
native_series = {name: series.groupby(level=0).mean().sort_index().rename(name)
                 for name, series in native_series.items()}
native_full = pd.concat(native_series, axis=1).sort_index()[MODEL_COLS]
display(pd.DataFrame({name: {'readings': len(series), 'finite': int(series.notna().sum()),
                            'missing': int(series.isna().sum())}
                      for name, series in native_series.items()}).T)
''')

md('''
### Original-timestamp plots, before modelling preprocessing

Plot each instrument on its own timestamps with shared time axes. Points are used
instead of joining missing intervals with lines. The first figure covers the separate
exports; the second shows the full existing merge and is explicitly labelled as such.
All observations are displayed, including extremes. No smoothing, interpolation,
normalization, outlier clipping or model fitting has occurred.
''')
code('''
def date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

def plot_variables(series_by_name, title, points=False):
    fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=True, layout='constrained')
    for ax, name in zip(axes, PLOT_ORDER):
        series = series_by_name[name]
        label, unit, colour = LABELS[name]
        if points:
            ax.scatter(series.index, series, s=1.2, alpha=0.45, color=colour, rasterized=True)
        else:
            ax.plot(series.index, series, linewidth=0.65, color=colour)
        ax.set(title=label, ylabel=unit)
        if name == TARGET:
            ax.axhline(0, color='grey', linewidth=0.6)
    date_axis(axes[-1])
    axes[-1].set_xlabel('Recorded timestamp (timezone not supplied)')
    fig.suptitle(title, fontsize=15)
    plt.show()
    plt.close(fig)

plot_variables(native_series, 'Instrument exports on their original timestamps', points=True)
plot_variables({name: full[name] for name in MODEL_COLS},
               'Full campaign: existing merge before this notebook’s preprocessing', points=True)
''')

md('''
## 3. Synchronize without inventing observations

For each variable, calculate the mean of finite readings in the same fixed, left-closed,
left-labelled **three-minute** bins anchored at midnight. One or more readings per
variable per bin is required. This aligns intervals, not instantaneous measurements:
within-bin timing differences can remain, up to the bin width. Missing bins stay `NaN`.

Compare the freshly aligned native exports with the full table over their shared period.
Differences can arise from the earlier merge's assignment/averaging; the comparison is
a diagnostic, not proof of the unknown earlier merge. Then choose the configured scope
and remove bins missing **any of the six model variables**. A uniform underlying grid
with excluded intervals is expected; retained rows need not be consecutive in time.

No time interpolation or moving average is used. Aggregation alone averages readings
inside each bin. The latest audio asks to remove missing intervals, and additional
smoothing would alter the target and introduce overlap between adjacent samples.
''')
code('''
def bin_series(series):
    resampler = series.resample(GRID, origin='start_day', closed='left', label='left')
    return resampler.mean(), resampler.count()

native_pairs = {name: bin_series(native_series[name]) for name in MODEL_COLS}
native_grid = pd.concat({name: pair[0] for name, pair in native_pairs.items()}, axis=1)
native_counts = pd.concat({name: pair[1] for name, pair in native_pairs.items()}, axis=1).fillna(0)
full_resampler = full[MODEL_COLS].resample(GRID, origin='start_day', closed='left', label='left')
full_grid = full_resampler.mean()
full_counts = full_resampler.count()

overlap_rows = []
for name in MODEL_COLS:
    common = pd.concat([native_grid[name].rename('Native'), full_grid[name].rename('Previous merge')],
                       axis=1).dropna()
    error = common['Previous merge'] - common['Native']
    overlap_rows.append({'Variable': name, 'Paired bins': len(common),
                         'Mean absolute difference': error.abs().mean(),
                         'Maximum absolute difference': error.abs().max()})
display(pd.DataFrame(overlap_rows).set_index('Variable'))

aligned = (full_grid if DATA_SCOPE == 'full_campaign' else native_grid).copy()
counts = (full_counts if DATA_SCOPE == 'full_campaign' else native_counts).copy()
complete = aligned[MODEL_COLS].notna().all(axis=1)
model_data = aligned.loc[complete, MODEL_COLS].copy()
rejected = aligned.index[~complete]
assert (counts.loc[complete, MODEL_COLS] >= 1).all().all()
assert np.isfinite(model_data.to_numpy()).all()
assert model_data.index.is_unique and model_data.index.is_monotonic_increasing
assert ((model_data.index - model_data.index.normalize()) % pd.Timedelta(GRID) == pd.Timedelta(0)).all()
assert len(model_data) > 100, 'Too few complete bins for meaningful validation.'

coverage = pd.DataFrame({'Observed bins': aligned.notna().sum(),
                         'Missing bins': aligned.isna().sum(),
                         'Coverage (%)': 100 * aligned.notna().mean()})
display(coverage.loc[PLOT_ORDER])
display(pd.Series({'Grid slots': len(aligned), 'Complete retained bins': len(model_data),
                   'Excluded incomplete bins': len(rejected), 'Interpolated bins': 0,
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
display(Markdown('**Largest excluded intervals**'))
display(gap_table.head(10).reset_index(drop=True))
''')

code('''
# Rows show per-variable availability before intersection; the last row is the model eligibility mask.
availability = np.vstack([aligned[PLOT_ORDER].notna().to_numpy().T, complete.to_numpy()])
fig, ax = plt.subplots(figsize=(13, 3.8), layout='constrained')
start_num = mdates.date2num(aligned.index[0])
end_num = mdates.date2num(aligned.index[-1] + pd.Timedelta(GRID))
ax.imshow(availability, aspect='auto', interpolation='nearest', cmap='Greys', vmin=0, vmax=1,
          extent=[start_num, end_num, len(PLOT_ORDER)+0.5, -0.5])
ax.set_yticks(range(7), [*PLOT_ORDER, 'All six present'])
ax.set_title(f'Coverage on the {GRID} grid: black = observed, white = missing ({DATA_SCOPE})')
ax.grid(False)
ax.xaxis_date()
date_axis(ax)
plt.show()
plt.close(fig)

# Restore missing grid positions for plotting only; never use dropna() to connect across outages.
plot_grid = model_data.reindex(aligned.index)
assert plot_grid.loc[~complete].isna().all().all()
plot_variables({name: plot_grid[name] for name in MODEL_COLS},
               f'After alignment: {GRID} means; incomplete intervals excluded; no interpolation')
''')

md('''
### Compare acquisition and alignment on the same time window

The next plot contrasts input readings with bin means for a representative early
three-day window. Its title states whether those readings came from native exports
or the existing full merge. Missing processed bins remain gaps. This checks what the
alignment changed without averaging across instrument outages.
''')
code('''
window_start = model_data.index.min().normalize() + pd.Timedelta('2D')
window_end = window_start + pd.Timedelta('3D')
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout='constrained')
for ax, name in zip(axes, ['Q_net', 'G', 'T_sky']):
    before = full[name] if DATA_SCOPE == 'full_campaign' else native_series[name]
    before = before.loc[window_start:window_end]
    after = plot_grid.loc[window_start:window_end, name]
    ax.scatter(before.index, before, color='#aaaaaa', s=5, alpha=0.6, label='Input readings')
    ax.plot(after.index, after, color=LABELS[name][2], linewidth=1.1, label=f'{GRID} bin mean')
    ax.set_ylabel(LABELS[name][1])
    ax.legend(loc='upper right')
date_axis(axes[-1])
fig.suptitle(f'Before and after alignment ({DATA_SCOPE}, {window_start:%d %b}–{window_end:%d %b})')
plt.show()
plt.close(fig)
''')

md('''
## 4. Data checks and descriptive relationships

Retain recorded heat-flux extremes: there is no documented rule permitting arbitrary
clipping to 0–100 W/m². Report them for investigation instead. Values at exactly
−50 °C suggest a sky-sensor floor, but the hardware range is not independently verified.
They remain recorded inputs, flagged below rather than replaced with invented values.
The count of invalid numeric values is separate from physically unusual finite values.

Daily averages and correlations describe the retained observations, not the missing
periods. Hours with more retained observations carry more weight. The timestamps have
no timezone metadata; hour-of-day plots use the recorded clock as supplied. Correlation
does not establish causality or substitute for held-out model evaluation.
''')
code('''
display(model_data.describe(percentiles=[0.01, 0.5, 0.99]).T)
input_for_checks = full[MODEL_COLS] if DATA_SCOPE == 'full_campaign' else native_full[MODEL_COLS]
flags = pd.Series({
    'Sky values exactly -50 °C (input readings)': int(input_for_checks['T_sky'].eq(-50).sum()),
    'RH outside 0–100% (input readings)': int((input_for_checks['RH'].lt(0) | input_for_checks['RH'].gt(100)).sum()),
    'Negative wind speed (input readings)': int(input_for_checks['v'].lt(0).sum()),
    'Negative solar irradiance (input readings)': int(input_for_checks['G'].lt(0).sum()),
    'Retained bins with |Q_net| > 100 W/m² (review, not automatic rejection)': int(model_data[TARGET].abs().gt(100).sum()),
}, name='Count')
display(flags.to_frame())
display(Markdown('**Ten largest recorded absolute target values after binning**'))
display(model_data.loc[model_data[TARGET].abs().nlargest(10).index])

correlation = model_data[MODEL_COLS].corr()
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
im = axes[0].imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xticks(range(6), MODEL_COLS, rotation=45, ha='right')
axes[0].set_yticks(range(6), MODEL_COLS)
axes[0].grid(False)
for row in range(6):
    for col in range(6):
        value = correlation.iloc[row, col]
        axes[0].text(col, row, f'{value:.2f}', ha='center', va='center',
                     color='white' if abs(value) > 0.65 else 'black')
axes[0].set_title('Pearson correlation: retained bins')
fig.colorbar(im, ax=axes[0], shrink=0.8)
axes[1].hist(model_data[TARGET], bins=100, color=LABELS[TARGET][2], alpha=0.85)
axes[1].set(xlabel=LABELS[TARGET][1], ylabel='Retained bins', title='Target distribution (no clipping)')
plt.show()
plt.close(fig)

hour_groups = model_data.groupby(model_data.index.hour)
hourly_means = hour_groups[[TARGET, 'G']].mean()
hourly_counts = hour_groups.size()
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout='constrained')
for ax, name in zip(axes[:2], [TARGET, 'G']):
    ax.plot(hourly_means.index, hourly_means[name], marker='o', color=LABELS[name][2])
    ax.set_ylabel(LABELS[name][1])
axes[2].bar(hourly_counts.index, hourly_counts, color='#8693a5')
axes[2].set(xlabel='Hour of recorded clock', ylabel='Retained bins', xticks=range(0, 24, 2))
fig.suptitle('Mean daily cycle and observation counts; excluded periods contribute no estimates')
plt.show()
plt.close(fig)
''')

md('''
## 5. Fixed training and validation splits

Use exactly the five atmospheric predictors, excluding timestamps, sample temperature,
heat-flux voltage and other target-derived columns.

- **Random 80/20:** a reproducible baseline. Nearby times can land in both sets, so
  its score describes interpolation among campaign conditions and can be optimistic.
- **Chronological 80/20:** the last 20% of retained bins form a separate later-period
  holdout. Remove the 24 hours immediately before that period from training as a guard
  against adjacent-time dependence. The date boundary is determined before fitting.

Each evaluation gets a fresh model and a fresh `MinMaxScaler`, fitted only on its own
training data. Normalization follows the audio, although Random Forest does not require
it. Do not clip held-out values outside the training range. The target is not normalized,
so errors remain in W/m². This is fixed baseline evaluation, not model selection on a
validation set. No hyperparameters are tuned using either holdout.
''')
code('''
X = model_data[FEATURES]
y = model_data[TARGET]
positions = np.arange(len(model_data))
random_train, random_valid = train_test_split(positions, test_size=VALIDATION_FRACTION,
                                             random_state=SEED, shuffle=True)
cut = int(np.floor((1 - VALIDATION_FRACTION) * len(model_data)))
chrono_valid = positions[cut:]
chronological_start = model_data.index[cut]
chrono_train = positions[model_data.index < chronological_start - CHRONOLOGICAL_GAP]
splits = {'Random 80/20': (np.sort(random_train), np.sort(random_valid)),
          'Chronological holdout': (chrono_train, chrono_valid)}
split_rows = []
for name, (train, valid) in splits.items():
    assert len(train) > 100 and len(valid) > 20
    assert np.intersect1d(train, valid).size == 0
    split_rows.append({'Evaluation': name, 'Training bins': len(train), 'Validation bins': len(valid),
                       'Unused gap bins': len(model_data)-len(train)-len(valid),
                       'Training first': X.index[train].min(), 'Training last': X.index[train].max(),
                       'Validation first': X.index[valid].min(), 'Validation last': X.index[valid].max()})
assert X.index[chrono_train].max() < chronological_start - CHRONOLOGICAL_GAP
display(pd.DataFrame(split_rows).set_index('Evaluation'))

fig, axes = plt.subplots(2, 1, figsize=(13, 4), sharex=True, layout='constrained')
for ax, (name, (train, valid)) in zip(axes, splits.items()):
    ax.scatter(X.index[train], np.zeros(len(train)), marker='|', s=18, label='Training', color='#2376b7')
    ax.scatter(X.index[valid], np.ones(len(valid)), marker='|', s=18, label='Validation', color='#dc8b12')
    ax.set(yticks=[0, 1], yticklabels=['Training', 'Validation'], title=name, ylim=(-0.5, 1.5))
date_axis(axes[-1])
plt.show()
plt.close(fig)
''')

md('''
## 6. Train a basic Random Forest and evaluate it

Use 100 trees with the standard untuned scikit-learn settings (`max_depth=None`,
`min_samples_leaf=1`, `max_features=1.0`, bootstrap sampling). Fix only the random seed
and use two CPU workers. A training-mean predictor provides a simple reference using
exactly the same training/validation indices.

RMSE penalizes large errors more strongly than MAE. R² is measured against variation
in the validation observations and can be negative. Training scores are shown to expose
overfitting, not presented as evidence of generalization. The presentation's Sample 1
MLP scores are not acceptance thresholds for this different campaign and model.
''')
code('''
def regression_metrics(observed, predicted):
    return {'RMSE (W/m²)': np.sqrt(mean_squared_error(observed, predicted)),
            'MAE (W/m²)': mean_absolute_error(observed, predicted),
            'R²': r2_score(observed, predicted),
            'Bias (prediction − measurement)': float(np.mean(np.asarray(predicted) - np.asarray(observed)))}

fitted_models, predictions, metric_rows = {}, {}, []
for name, (train, valid) in splits.items():
    pipeline = Pipeline([
        ('scale', MinMaxScaler()),
        ('forest', RandomForestRegressor(n_estimators=100, max_depth=None,
                                         min_samples_leaf=1, max_features=1.0,
                                         bootstrap=True, random_state=SEED, n_jobs=2)),
    ])
    pipeline.fit(X.iloc[train], y.iloc[train])
    predicted = pipeline.predict(X.iloc[valid])
    baseline = DummyRegressor(strategy='mean').fit(X.iloc[train], y.iloc[train])
    baseline_predicted = baseline.predict(X.iloc[valid])
    assert np.isfinite(predicted).all()
    scaler = pipeline.named_steps['scale']
    assert np.allclose(scaler.data_min_, X.iloc[train].min())
    assert np.allclose(scaler.data_max_, X.iloc[train].max())
    fitted_models[name] = pipeline
    predictions[name] = pd.DataFrame({'Measured': y.iloc[valid], 'Predicted': predicted,
                                      'Training-mean baseline': baseline_predicted}, index=y.iloc[valid].index)
    predictions[name]['Residual (prediction − measurement)'] = predicted - y.iloc[valid]
    for model_name, partition, observed, estimate in [
        ('Random Forest', 'Validation', y.iloc[valid], predicted),
        ('Training-mean baseline', 'Validation', y.iloc[valid], baseline_predicted),
        ('Random Forest', 'Training', y.iloc[train], pipeline.predict(X.iloc[train])),
    ]:
        metric_rows.append({'Evaluation': name, 'Model': model_name, 'Partition': partition,
                            'N': len(observed), **regression_metrics(observed, estimate)})

metrics = pd.DataFrame(metric_rows)
display(Markdown('**Held-out results (the actual performance estimates)**'))
display(metrics.loc[metrics.Partition.eq('Validation')].set_index(['Evaluation', 'Model']).drop(columns='Partition').round(4))
display(Markdown('**Training results (overfitting diagnostic only)**'))
display(metrics.loc[metrics.Partition.eq('Training')].set_index('Evaluation').drop(columns=['Model', 'Partition']).round(4))
''')

md('''
### Measured validation values versus predictions

These are the parity plots requested in the audio and illustrated on presentation
slide 13. Only held-out predictions are plotted. Axes have the same units and scale;
the dashed line is perfect agreement. All validation observations, including extreme
errors, are included. Each annotation reports the corresponding held-out metrics.
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
for ax, (name, result) in zip(axes, predictions.items()):
    ax.scatter(result['Measured'], result['Predicted'], s=9, alpha=0.32, color='#8e2548', rasterized=True)
    lower = result[['Measured', 'Predicted']].min().min()
    upper = result[['Measured', 'Predicted']].max().max()
    padding = 0.04 * max(upper-lower, 1)
    bounds = (lower-padding, upper+padding)
    ax.plot(bounds, bounds, '--', color='black', linewidth=1, label='1:1 agreement')
    ax.set(xlim=bounds, ylim=bounds, xlabel='Measured Q_net [W/m²]',
           ylabel='Predicted Q_net [W/m²]', title=name)
    ax.set_aspect('equal', adjustable='box')
    values = regression_metrics(result['Measured'], result['Predicted'])
    annotation = (f"N = {len(result):,}\\nRMSE = {values['RMSE (W/m²)']:.2f} W/m²\\n"
                  f"MAE = {values['MAE (W/m²)']:.2f} W/m²\\nR² = {values['R²']:.3f}")
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes, va='top',
            bbox={'facecolor': 'white', 'alpha': 0.9, 'edgecolor': '#cccccc'})
    ax.legend(loc='lower right')
fig.suptitle('Random Forest: held-out measurements versus predictions', fontsize=14)
plt.show()
plt.close(fig)
''')

md('''
## 7. Check errors and reliance on the inputs

Inspect the later-period holdout over time, using points so that absent intervals are
not connected. Permutation importance below measures the increase in validation RMSE
when one input is shuffled. It describes this fitted model's reliance on the input,
not causal influence or an intrinsic material property. Correlated inputs can share
or obscure importance. This diagnostic does not change the fitted model or its settings.
''')
code('''
later = predictions['Chronological holdout']
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout='constrained')
axes[0].scatter(later.index, later['Measured'], s=5, alpha=0.7, color='#202020', label='Measured')
axes[0].scatter(later.index, later['Predicted'], s=5, alpha=0.5, color='#c83d4b', label='Predicted')
axes[0].set(ylabel=LABELS[TARGET][1], title='Chronological holdout: measured and predicted heat flux')
axes[0].legend()
axes[1].scatter(later.index, later['Residual (prediction − measurement)'], s=5, alpha=0.5, color='#2376b7')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_ylabel('Prediction − measurement [W/m²]')
date_axis(axes[1])
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
for name, result in predictions.items():
    residuals = result['Residual (prediction − measurement)']
    axes[0].hist(residuals, bins=60, alpha=0.55, density=True, label=name)
    axes[1].scatter(result['Predicted'], residuals, s=5, alpha=0.25, label=name)
axes[0].set(xlabel='Prediction − measurement [W/m²]', ylabel='Density', title='Held-out residual distribution')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set(xlabel='Predicted Q_net [W/m²]', ylabel='Prediction − measurement [W/m²]', title='Residuals versus predictions')
axes[0].legend()
axes[1].legend()
plt.show()
plt.close(fig)

train, valid = splits['Chronological holdout']
importance = permutation_importance(fitted_models['Chronological holdout'], X.iloc[valid], y.iloc[valid],
                                    scoring='neg_root_mean_squared_error', n_repeats=5,
                                    random_state=SEED, n_jobs=1)
importance_table = pd.DataFrame({'Feature': FEATURES, 'RMSE increase (W/m²)': importance.importances_mean,
                                 'Shuffle SD (W/m²)': importance.importances_std}).sort_values('RMSE increase (W/m²)')
display(importance_table.set_index('Feature').round(4))
fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
ax.barh(importance_table.Feature, importance_table['RMSE increase (W/m²)'],
        xerr=importance_table['Shuffle SD (W/m²)'], color='#2376b7', alpha=0.8, capsize=3)
ax.axvline(0, color='black', linewidth=0.7)
ax.set(xlabel='Increase in held-out RMSE after shuffling [W/m²]',
       title='Permutation importance on the chronological holdout (5 shuffles)')
plt.show()
plt.close(fig)

range_check = pd.DataFrame({'Train min': X.iloc[train].min(), 'Train max': X.iloc[train].max(),
                            'Validation min': X.iloc[valid].min(), 'Validation max': X.iloc[valid].max()})
range_check['Validation outside training range (%)'] = 100 * (
    X.iloc[valid].lt(X.iloc[train].min()) | X.iloc[valid].gt(X.iloc[train].max())).mean()
display(Markdown('**Input-range shift in the later-period holdout**'))
display(range_check.round(3))
''')

md('''
## 8. Results, interpretation and reusable prediction function

The next summary is generated from the executed results. A high random-split score
does not override a weaker chronological result. Neither metric establishes performance
on a different material, instrument calibration or future campaign. The source-alignment
and heat-flux calibration limitations described above remain even when code checks pass.

After evaluating the fixed baseline, fit a separate model on all retained bins for
possible reuse. This final fit has **no independent test score**: the reported scores
belong to the earlier held-out evaluations. The reusable function expects the same
three-minute mean inputs, names, units, material and measurement convention. It returns
recorded-sign heat flux. It is not a calibrated cooling-power certification.
''')
code('''
validation_metrics = metrics.loc[metrics.Partition.eq('Validation')].set_index(['Evaluation', 'Model'])
summary_lines = [f'### Executed baseline result: {DATA_SCOPE}',
                 f'- Retained **{len(model_data):,}** observed {GRID} bins from '
                 f'**{model_data.index.min():%d %b %Y}** to **{model_data.index.max():%d %b %Y}**.',
                 f'- Excluded **{len(rejected):,}** incomplete grid intervals. **Zero values were interpolated.**']
for name in splits:
    forest_result = validation_metrics.loc[(name, 'Random Forest')]
    baseline_result = validation_metrics.loc[(name, 'Training-mean baseline')]
    improvement = 100 * (1 - forest_result['RMSE (W/m²)'] / baseline_result['RMSE (W/m²)'])
    summary_lines.append(f"- **{name}:** RMSE **{forest_result['RMSE (W/m²)']:.2f} W/m²**, "
                         f"MAE **{forest_result['MAE (W/m²)']:.2f} W/m²**, R² **{forest_result['R²']:.3f}**. "
                         f"RMSE change versus the training-mean baseline: **{improvement:.1f}% reduction** "
                         '(a negative reduction means worse performance).')
random_score = validation_metrics.loc[('Random 80/20', 'Random Forest'), 'R²']
later_score = validation_metrics.loc[('Chronological holdout', 'Random Forest'), 'R²']
if later_score < random_score:
    summary_lines.append('- Later-period performance is weaker than random-split performance; '
                         'use the chronological result when discussing prediction at later dates.')
if later_score < 0:
    summary_lines.append('- Chronological R² is negative: the model has not demonstrated useful '
                         'later-period generalization on this split.')
if DATA_SCOPE == 'full_campaign':
    summary_lines.append('- Full-campaign results remain conditional on the earlier merge; native '
                         'instrument alignment after 15 May cannot be independently audited with these files.')
summary_lines.append('- Verify the inherited heat-flux calibration/sign and investigate sensor-floor '
                     'values and extreme residuals before claiming physical cooling performance.')
display(Markdown('\\n'.join(summary_lines)))

final_model = Pipeline([
    ('scale', MinMaxScaler()),
    ('forest', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1,
                                     max_features=1.0, bootstrap=True, random_state=SEED, n_jobs=2)),
]).fit(X, y)

def predict_net_heat_flux(atmospheric_means):
    """Predict recorded-sign W/m² from a DataFrame of the five bin-mean inputs."""
    if not isinstance(atmospheric_means, pd.DataFrame):
        raise TypeError('Pass a pandas DataFrame with the five named atmospheric inputs.')
    values = atmospheric_means.loc[:, FEATURES].apply(pd.to_numeric, errors='raise')
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError('Provide at least one row of finite, non-missing input values.')
    outside = values.lt(X.min()) | values.gt(X.max())
    return pd.DataFrame({'Predicted Q_net (W/m²)': final_model.predict(values),
                         'Outside fitted input range': outside.any(axis=1)}, index=values.index)

display(Markdown('**Function demonstration on three existing input rows (not a new validation score):**'))
display(predict_net_heat_flux(X.tail(3)))
''')

md('''
### Reproducibility checks

These checks ensure that the notebook followed its stated processing and evaluation
rules. They do not certify sensor calibration or the unavailable earlier merge.
All figures are notebook outputs. No separate image, model or data file is required
to view this executed report; rerunning requires the four original input CSVs.
''')
code('''
assert model_data.equals(aligned.loc[aligned[MODEL_COLS].notna().all(axis=1), MODEL_COLS])
assert model_data.index.intersection(rejected).empty
assert (counts.loc[model_data.index, MODEL_COLS] >= 1).all().all()
assert list(X.columns) == FEATURES and TARGET not in X.columns
for name, (train, valid) in splits.items():
    result = predictions[name]
    assert result.index.equals(y.iloc[valid].index)
    assert np.allclose(result.Measured, y.iloc[valid])
    independent_rmse = np.sqrt(np.mean((result.Predicted-result.Measured)**2))
    assert np.isclose(independent_rmse, validation_metrics.loc[(name, 'Random Forest'), 'RMSE (W/m²)'])
    assert fitted_models[name].named_steps['scale'].n_samples_seen_ == len(train)
for name, filename in source_names.items():
    assert hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest() == source_inventory.loc[name, 'SHA256']
print('Checks passed: inputs unchanged; observed bins only; exclusions preserved; disjoint splits;')
print('training-only normalization; no target-derived inputs; aligned held-out predictions; verified RMSE.')
print('All figures and results are embedded in main_5.ipynb. No separate figure files were written.')
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.9'},
})
nbf.validate(nb)
DEST.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, DEST)
print(f'Created {DEST} with {len(cells)} cells')
