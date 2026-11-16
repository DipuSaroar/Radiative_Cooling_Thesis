from pathlib import Path
import textwrap
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'Dipu_Bhai_Thesis/code/main_5.ipynb'
nb = nbformat.read(path, as_version=4)
assert not any('Individual time-series figures' in c.source for c in nb.cells)

def md(source):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())

def code(source):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())

individual = [md('''
### Individual time-series figures

These six full-size figures correspond to `Q_net_vs_time`, `G_vs_time`,
`T_amb_vs_time`, `T_sky_vs_time`, `RH_vs_time` and `v_vs_time` in the earlier
`figures_main_3` collection. They are regenerated from this notebook's selected
dataset on its three-minute grid. Missing intervals remain visible gaps, and the
recorded heat-flux sign is preserved. The coverage and combined six-variable
figures appear immediately above. Every figure is displayed and stored inline;
no separate PNG or PDF is written.
'''), code('''
for name in PLOT_ORDER:
    title, unit, colour = LABELS[name]
    fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
    ax.plot(plot_grid.index, plot_grid[name], color=colour, linewidth=0.8)
    if name == TARGET:
        ax.axhline(0, color='grey', linewidth=0.7)
    ax.set(title=f'{title} vs time', ylabel=unit,
           xlabel='Recorded timestamp (timezone not supplied)')
    date_axis(ax)
    ax.text(0.01, 0.98, f'{GRID} means; incomplete intervals excluded; no interpolation',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    plt.show()
    plt.close(fig)
''')]

comparison = [md('''
### Mean daily cycle: net heat flux and solar irradiance

This reproduces the two-axis layout of `02_mean_daily_cycle` using only the retained
observations. Read heat flux against the left axis and solar irradiance against the
right axis; the heights of the two curves are not directly comparable. The separate
panels and observation counts above provide the same information without dual axes.
'''), code('''
daily_cycle = model_data.groupby(model_data.index.hour)[[TARGET, 'G']].mean()
fig, flux_axis = plt.subplots(figsize=(12, 5), layout='constrained')
solar_axis = flux_axis.twinx()
flux_line, = flux_axis.plot(daily_cycle.index, daily_cycle[TARGET], marker='o',
                            linewidth=2, markersize=4, color=LABELS[TARGET][2],
                            label='Net heat flux (left axis)')
solar_line, = solar_axis.plot(daily_cycle.index, daily_cycle['G'], marker='s',
                              linewidth=2, markersize=4, color=LABELS['G'][2],
                              label='Solar irradiance (right axis)')
flux_axis.axhline(0, color='grey', linewidth=0.7)
flux_axis.set(xlabel='Hour of recorded clock', ylabel=LABELS[TARGET][1],
              xticks=range(0, 24, 2), title='Mean daily cycle: net heat flux and solar irradiance')
solar_axis.set_ylabel(LABELS['G'][1], color=LABELS['G'][2])
solar_axis.tick_params(axis='y', colors=LABELS['G'][2])
solar_axis.spines['right'].set_visible(True)
solar_axis.grid(False)
flux_axis.legend(handles=[flux_line, solar_line], loc='upper left')
plt.show()
plt.close(fig)
'''), md('''
### Standalone correlation heatmap

This is the full-size counterpart of `03_correlation_heatmap`. It uses the same
retained-bin Pearson correlations as the diagnostic panel above. Correlations are
descriptive associations, not causal effects or measures of prediction accuracy.
'''), code('''
standalone_correlation = model_data[MODEL_COLS].corr()
fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
colour_map = ax.imshow(standalone_correlation, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(MODEL_COLS)), MODEL_COLS, rotation=45, ha='right')
ax.set_yticks(range(len(MODEL_COLS)), MODEL_COLS)
for row in range(len(MODEL_COLS)):
    for column in range(len(MODEL_COLS)):
        value = standalone_correlation.iloc[row, column]
        ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=11,
                color='white' if abs(value) > 0.65 else 'black')
ax.set_title('Pearson correlation of model variables: retained observations')
ax.grid(False)
fig.colorbar(colour_map, ax=ax, shrink=0.85, label='Pearson correlation')
plt.show()
plt.close(fig)
''')]

# Insert from the later position first so the earlier position remains stable.
index = next(i for i,c in enumerate(nb.cells) if c.source.startswith('## 5. Fixed training'))
nb.cells[index:index] = comparison
index = next(i for i,c in enumerate(nb.cells) if c.source.startswith('### Compare acquisition'))
nb.cells[index:index] = individual
nbformat.validate(nb)
nbformat.write(nb, path)
print('Added six individual time series, a dual-axis daily cycle and a standalone heatmap.')
