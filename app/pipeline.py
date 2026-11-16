"""Processing and modelling pipeline shared by every page of the web app.

The logic mirrors the thesis notebooks so the app and the report cannot disagree:

* raw parsing  -> ``code/data_cleaning.ipynb`` (RAFT, weather station, IR thermometer)
* processing   -> ``code/main.ipynb`` sections 2-3 (20-sample centred moving average,
                  3-minute bins, incomplete bins deleted, nothing interpolated)
* modelling    -> ``code/main.ipynb`` sections 5-7 (untuned Random Forest, random 80/20
                  split and chronological holdout with a 24-hour purge)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw_data"

# --- Settings (identical to code/main.ipynb) -------------------------------------------
GRID = "3min"
MOVING_WINDOW = 20
MAX_SOURCE_GAP = pd.Timedelta("3min")
SEED = 42
VALIDATION_FRACTION = 0.20
CHRONOLOGICAL_GAP = pd.Timedelta("1D")
HEAT_FLUX_SENSITIVITY_V = 60e-6  # V per (W/m²), inherited calibration
HFP01_MAX_VOLTAGE_V = 0.12  # |reading| above this is a microvolt reading (original reader rule)

FEATURES = ["T_amb", "RH", "v", "G", "T_sky"]
TARGET = "Q_net"
MODEL_COLS = FEATURES + [TARGET]
PLOT_ORDER = [TARGET, "G", "T_amb", "T_sky", "RH", "v"]
LABELS = {
    "Q_net": ("Net heat flux", "W/m²", "#202020"),
    "G": ("Solar irradiance", "W/m²", "#dc8b12"),
    "T_amb": ("Ambient temperature", "°C", "#2376b7"),
    "T_sky": ("Sky temperature", "°C", "#c83d4b"),
    "RH": ("Relative humidity", "%", "#329261"),
    "v": ("Wind speed", "m/s", "#8661b3"),
}


# =========================================================================================
# 1. Raw file parsing (same rules as code/data_cleaning.ipynb)
# =========================================================================================

RAFT_LINE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})\s*$")
METEO_COLUMNS = ["TIME", "TEMP", "DEWPOINT", "RH", "PRESS", "SPEED", "DIR", "SOLARRAD", "CHECKSUM"]
DAT_META = ["Date", "Time", "Unit", "Resolution", "Values", "SnapshotTrigger", "MeasurementTrigger"]
IR_SIMPLE_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+([+-]?\d+,\d+)\s*$")
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

FILE_KINDS = {
    "raft": "RAFT heat-flux logger",
    "meteo": "Weather station",
    "ir_dat": "IR thermometer (detailed .dat)",
    "ir_txt": "IR thermometer (simple .txt)",
    "csv": "Table (CSV with timestamp and variables)",
}


def _lines(data: bytes, encoding: str = "latin-1") -> list[str]:
    return re.split(r"\r\n|\r|\n", data.decode(encoding, errors="replace"))


def classify_file(name: str, data: bytes) -> str | None:
    """Recognise a file by its content first, then by its name."""
    head = data[:4000].decode("latin-1", errors="replace")
    lower = name.lower()
    if "[Connect DataFile]" in head:
        return "ir_dat"
    if head.lstrip().lower().startswith("data ora temperatura"):
        return "ir_txt"
    if head.startswith("TIME,TEMP") or lower.endswith(".log"):
        return "meteo"
    if "Heat-flux" in head or ("raft" in lower and lower.endswith(".txt")):
        return "raft"
    if lower.endswith(".csv"):
        return "csv"
    if "ir" in lower and lower.endswith(".txt"):
        return "ir_txt"
    return None


def parse_raft(name: str, data: bytes) -> pd.DataFrame:
    rows = []
    for line in _lines(data):
        match = RAFT_LINE.match(line)
        if match:
            heat_flux, pt100, date, time = match.groups()
            rows.append((heat_flux.replace(",", "."), pt100.replace(",", "."), date, time, name))
    return pd.DataFrame(rows, columns=["HEAT_FLUX_V", "PT100_VENT_OHM", "DATE", "TIME", "SOURCE_FILE"])


def parse_meteo(name: str, data: bytes) -> pd.DataFrame:
    rows = []
    for line in _lines(data)[1:]:
        if not line.strip():
            continue
        fields = [f.strip() for f in re.sub(r"[\x00-\x1f]", "", line).split(",")]
        if len(fields) == len(METEO_COLUMNS):
            rows.append(fields + [name])
    return pd.DataFrame(rows, columns=METEO_COLUMNS + ["SOURCE_FILE"])


def _ps_name_order(name: str) -> str:
    return name.lower().replace("-", "")


def parse_ir(files: list[tuple[str, bytes, str]]) -> pd.DataFrame:
    """Merge every IR file, keeping repeated readings from overlapping exports once."""
    detailed: dict = {}
    simple: dict = {}
    for name, data, kind in sorted(files, key=lambda f: _ps_name_order(f[0])):
        lines = data.decode("cp1252", errors="replace").splitlines()
        if kind == "ir_dat":
            meta = {k: next((l[len(k) + 1:].strip() for l in lines if l.startswith(k + ":")), "")
                    for k in DAT_META}
            start = datetime.strptime(f"{meta['Date']} {meta['Time']}", "%d/%m/%Y %H:%M:%S")
            for line in lines:
                if not re.match(r"^\d{3}:\d{2}:\d{2},\d{3}\t", line):
                    continue
                parts = line.split("\t")
                if len(parts) != 17:
                    continue
                key = (meta["Date"], meta["Time"], line)
                if key in detailed:
                    continue
                h, m, s, ms = map(int, re.match(r"^(\d{3}):(\d{2}):(\d{2}),(\d{3})$", parts[0]).groups())
                absolute = re.match(r"^\d{2}:\d{2}:\d{2}:(\d{3})$", parts[14])
                stamp = start + timedelta(hours=h, minutes=m, seconds=s,
                                          milliseconds=ms + (int(absolute.group(1)) if absolute else 0))
                detailed[key] = (stamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "DETAILED_DAT",
                                 parts[1].strip().replace(",", "."))
        else:
            for line in lines:
                match = IR_SIMPLE_LINE.match(line)
                if match:
                    key = (f"{match.group(1)} {match.group(2)}.000", match.group(3).replace(",", "."))
                    simple.setdefault(key, (key[0], "SIMPLE_TXT", key[1]))
    rows = list(detailed.values()) + list(simple.values())
    frame = pd.DataFrame(rows, columns=["TIMESTAMP", "RECORD_TYPE", "T_SKY"])
    return frame.sort_values(["TIMESTAMP", "RECORD_TYPE"], kind="stable").reset_index(drop=True)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _dedupe(series: pd.Series, name: str) -> pd.Series:
    series = series.sort_index(kind="stable")
    return series.loc[~series.index.duplicated(keep="first")].rename(name)


@dataclass
class ReadReport:
    files: pd.DataFrame  # one row per uploaded file
    readings: pd.DataFrame  # one row per variable


def read_native(files: list[tuple[str, bytes]]) -> tuple[dict[str, pd.Series], ReadReport]:
    """Turn raw instrument files (or CSV tables) into one time series per variable."""
    file_rows, raft_parts, meteo_parts, ir_files, csv_parts = [], [], [], [], []
    for name, data in files:
        kind = classify_file(name, data)
        rows = 0
        if kind == "raft":
            part = parse_raft(name, data)
            raft_parts.append(part)
            rows = len(part)
        elif kind == "meteo":
            part = parse_meteo(name, data)
            meteo_parts.append(part)
            rows = len(part)
        elif kind in ("ir_dat", "ir_txt"):
            ir_files.append((name, data, kind))
            text = data.decode("cp1252", errors="replace").splitlines()
            pattern = r"^\d{3}:\d{2}:\d{2},\d{3}\t" if kind == "ir_dat" else IR_SIMPLE_LINE.pattern
            rows = sum(1 for line in text if re.match(pattern, line))
        elif kind == "csv":
            part = pd.read_csv(io.BytesIO(data))
            part.columns = [c.strip() for c in part.columns]
            time_col = next((c for c in part.columns if c.lower() in ("timestamp", "time", "datetime")), None)
            if time_col is None:
                kind = None
            else:
                part.index = pd.to_datetime(part.pop(time_col), errors="coerce")
                part = part.loc[part.index.notna()]
                csv_parts.append(part)
                rows = len(part)
        file_rows.append({"File": name, "Recognised as": FILE_KINDS.get(kind, "Not recognised (ignored)"),
                          "Rows read": rows})

    series: dict[str, list[pd.Series]] = {c: [] for c in MODEL_COLS}
    if raft_parts:
        raft = pd.concat(raft_parts, ignore_index=True)
        raft.index = pd.to_datetime(raft["DATE"].str.strip() + " " + raft["TIME"].str.strip(),
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
        raft = raft.loc[raft.index.notna()].sort_index(kind="stable")
        raw = _numeric(raft["HEAT_FLUX_V"])
        volts = raw.where(raw.abs() <= HFP01_MAX_VOLTAGE_V, raw * 1e-6)
        series[TARGET].append(volts / HEAT_FLUX_SENSITIVITY_V)
    if meteo_parts:
        meteo = pd.concat(meteo_parts, ignore_index=True)
        parts = meteo["TIME"].str.extract(r"^\w+ (\d{1,2}) (\w+) (\d{4}) (\d{2}:\d{2}:\d{2})$")
        meteo.index = pd.to_datetime(parts[2] + "-" + parts[1].map(MONTHS).astype("Int64").astype(str).str.zfill(2)
                                     + "-" + parts[0].str.zfill(2) + " " + parts[3], errors="coerce")
        meteo = meteo.loc[meteo.index.notna()].sort_index(kind="stable")
        for original, name in {"TEMP": "T_amb", "RH": "RH", "SPEED": "v", "SOLARRAD": "G"}.items():
            series[name].append(_numeric(meteo[original]))
    if ir_files:
        ir = parse_ir(ir_files)
        ir.index = pd.to_datetime(ir["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S.%f")
        series["T_sky"].append(_numeric(ir["T_SKY"]).sort_index(kind="stable"))
    for part in csv_parts:
        for column in MODEL_COLS:
            if column in part.columns:
                series[column].append(_numeric(part[column]))

    native = {name: _dedupe(pd.concat(parts), name) for name, parts in series.items() if parts}
    readings = pd.DataFrame([
        {"Variable": name, "Readings": len(native[name]) if name in native else 0,
         "Valid readings": int(native[name].notna().sum()) if name in native else 0,
         "First": native[name].index.min() if name in native else pd.NaT,
         "Last": native[name].index.max() if name in native else pd.NaT}
        for name in PLOT_ORDER]).set_index("Variable")
    return native, ReadReport(pd.DataFrame(file_rows), readings)


def combine_native(*sources: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Join several sets of readings; on identical timestamps the earlier source wins."""
    names = {n for s in sources for n in s}
    return {n: _dedupe(pd.concat([s[n] for s in sources if n in s]), n) for n in names}


def bundled_raw_files() -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(RAW_DIR.iterdir()) if p.is_file()]


# =========================================================================================
# 2. Processing (code/main.ipynb section 3)
# =========================================================================================

@dataclass
class Processed:
    model_data: pd.DataFrame  # retained bins, physical units, no missing values
    unsmoothed: pd.DataFrame  # plain 3-min means on the same retained bins
    support: pd.DataFrame  # first/last raw reading behind each retained bin
    excluded_runs: pd.DataFrame  # deleted intervals
    summary: dict = field(default_factory=dict)


def _bin(series: pd.Series):
    resampler = series.resample(GRID, origin="start_day", closed="left", label="left")
    return resampler.mean(), resampler.count()


def _smooth(series: pd.Series, eligible_bins: pd.Series) -> pd.DataFrame:
    """MATLAB-style even-window centred moving average inside uninterrupted segments."""
    window = MOVING_WINDOW
    bins = series.index.floor(GRID)
    eligible = pd.Series(eligible_bins.reindex(bins, fill_value=False).to_numpy(), index=series.index)
    valid = series.notna() & eligible
    delta = series.index.to_series().diff()
    bin_delta = pd.Series(bins, index=series.index).diff()
    boundaries = (~valid | ~valid.shift(fill_value=False) |
                  delta.gt(MAX_SOURCE_GAP) | bin_delta.gt(pd.Timedelta(GRID)))
    groups = boundaries.cumsum()
    result = pd.DataFrame(index=series.index, columns=["value"], dtype=float)
    result["support_start"] = pd.NaT
    result["support_end"] = pd.NaT
    for _, segment in series.loc[valid].groupby(groups.loc[valid]):
        size = len(segment)
        position = np.arange(size)
        left = np.maximum(0, position - window // 2)
        right = np.minimum(size, position + window // 2)
        result.loc[segment.index, "value"] = segment.rolling(window, center=True, min_periods=1).mean()
        result.loc[segment.index, "support_start"] = segment.index[left].to_numpy()
        result.loc[segment.index, "support_end"] = segment.index[right - 1].to_numpy()
    return result


def _runs(mask: pd.Series) -> pd.DataFrame:
    runs = []
    for _, segment in mask.groupby(mask.ne(mask.shift()).cumsum()):
        if segment.iloc[0]:
            runs.append({"Start": segment.index[0], "End": segment.index[-1] + pd.Timedelta(GRID),
                         "Deleted bins": len(segment),
                         "Hours": len(segment) * pd.Timedelta(GRID).total_seconds() / 3600})
    return pd.DataFrame(runs, columns=["Start", "End", "Deleted bins", "Hours"])


class PipelineError(ValueError):
    """Raised when the uploaded data cannot produce a modelling table."""


def process(native: dict[str, pd.Series]) -> Processed:
    missing = [LABELS[n][0] for n in MODEL_COLS if n not in native or native[n].notna().sum() == 0]
    if missing:
        raise PipelineError("No readings for: " + ", ".join(missing) +
                            ". Upload files from all three instruments, or combine with the campaign data.")
    native = {n: native[n] for n in MODEL_COLS}
    pairs = {n: _bin(native[n]) for n in MODEL_COLS}
    grid = pd.concat({n: p[0] for n, p in pairs.items()}, axis=1)
    overlap_start = max(s.index.min() for s in native.values())
    overlap_end = min(s.index.max() for s in native.values())
    if overlap_start >= overlap_end:
        raise PipelineError("The instruments' recording periods do not overlap, so no complete bins exist.")
    common = grid.index[(grid.index >= overlap_start.ceil(GRID)) &
                        (grid.index + pd.Timedelta(GRID) <= overlap_end)]
    raw_aligned = grid.reindex(common)[MODEL_COLS]
    complete = raw_aligned.notna().all(axis=1)
    if complete.sum() < 10:
        raise PipelineError("Fewer than 10 complete 3-minute bins: every variable must be recorded at the same time.")

    smoothed = {n: _smooth(native[n], complete) for n in MODEL_COLS}
    aligned = pd.concat({n: _bin(t["value"])[0] for n, t in smoothed.items()}, axis=1).reindex(common)[MODEL_COLS]
    model_data = aligned.loc[complete].copy()
    model_data.index.name = "timestamp"
    starts = pd.concat({n: t["support_start"].resample(GRID).min() for n, t in smoothed.items()}, axis=1)
    ends = pd.concat({n: t["support_end"].resample(GRID).max() for n, t in smoothed.items()}, axis=1)
    support = pd.DataFrame({"source_start": starts.reindex(model_data.index).min(axis=1),
                            "source_end": ends.reindex(model_data.index).max(axis=1)})
    excluded = _runs(~complete)
    summary = {
        "retained_bins": len(model_data),
        "grid_slots": len(aligned),
        "excluded_bins": int((~complete).sum()),
        "excluded_periods": len(excluded),
        "first": model_data.index.min(),
        "last": model_data.index.max(),
        "coverage_pct": 100 * complete.mean(),
        "native_readings": {n: int(native[n].notna().sum()) for n in MODEL_COLS},
    }
    return Processed(model_data, raw_aligned.loc[complete], support, excluded, summary)


# =========================================================================================
# 3. Model (code/main.ipynb sections 5-7)
# =========================================================================================

def new_forest() -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("forest", RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1,
                                         max_features=1.0, bootstrap=True, random_state=SEED, n_jobs=-1)),
    ])


def regression_metrics(observed, predicted) -> dict:
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    return {"RMSE (W/m²)": float(np.sqrt(mean_squared_error(observed, predicted))),
            "MAE (W/m²)": float(mean_absolute_error(observed, predicted)),
            "R²": float(r2_score(observed, predicted)),
            "Bias (W/m²)": float(np.mean(predicted - observed))}


@dataclass
class Evaluation:
    metrics: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    importance: pd.DataFrame
    splits: pd.DataFrame
    chronological_disjoint: bool


def make_splits(model_data: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    positions = np.arange(len(model_data))
    random_train, random_valid = train_test_split(positions, test_size=VALIDATION_FRACTION,
                                                  random_state=SEED, shuffle=True)
    cut = int(np.floor((1 - VALIDATION_FRACTION) * len(model_data)))
    start = model_data.index[cut]
    chrono_train = positions[model_data.index < start - CHRONOLOGICAL_GAP]
    return {"Random 80/20": (np.sort(random_train), np.sort(random_valid)),
            "Chronological holdout": (chrono_train, positions[cut:])}


def evaluate(processed: Processed) -> Evaluation:
    data = processed.model_data
    if len(data) < 200:
        raise PipelineError("At least 200 retained bins (10 hours) are needed to train and validate the model.")
    X, y = data[FEATURES], data[TARGET]
    splits = make_splits(data)
    rows, predictions, split_rows, fitted = [], {}, [], {}
    for name, (train, valid) in splits.items():
        if len(train) < 100 or len(valid) < 20:
            continue
        model = new_forest().fit(X.iloc[train], y.iloc[train])
        fitted[name] = model
        predicted = model.predict(X.iloc[valid])
        baseline = DummyRegressor(strategy="mean").fit(X.iloc[train], y.iloc[train]).predict(X.iloc[valid])
        predictions[name] = pd.DataFrame({"Measured": y.iloc[valid], "Predicted": predicted,
                                          "Baseline": baseline}, index=y.iloc[valid].index)
        predictions[name]["Residual"] = predicted - y.iloc[valid]
        for model_name, partition, observed, estimate in [
            ("Random Forest", "Validation", y.iloc[valid], predicted),
            ("Training-mean baseline", "Validation", y.iloc[valid], baseline),
            ("Random Forest", "Training", y.iloc[train], model.predict(X.iloc[train])),
        ]:
            rows.append({"Evaluation": name, "Model": model_name, "Partition": partition,
                         "N": len(observed), **regression_metrics(observed, estimate)})
        split_rows.append({"Evaluation": name, "Training bins": len(train), "Validation bins": len(valid),
                           "Validation from": X.index[valid].min(), "Validation to": X.index[valid].max()})
    if not predictions:
        raise PipelineError("The data are too short for a chronological split with a 24-hour purge.")

    disjoint = False
    importance = pd.DataFrame(columns=["Feature", "RMSE increase (W/m²)", "Spread (W/m²)"])
    focus = "Chronological holdout" if "Chronological holdout" in fitted else next(iter(fitted))
    train, valid = splits[focus]
    if focus == "Chronological holdout":
        disjoint = bool(processed.support.iloc[train].source_end.max() <
                        processed.support.iloc[valid].source_start.min())
    result = permutation_importance(fitted[focus], X.iloc[valid], y.iloc[valid],
                                    scoring="neg_root_mean_squared_error", n_repeats=5,
                                    random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"Feature": FEATURES, "RMSE increase (W/m²)": result.importances_mean,
                               "Spread (W/m²)": result.importances_std}
                              ).sort_values("RMSE increase (W/m²)", ascending=False)
    return Evaluation(pd.DataFrame(rows), predictions, importance,
                      pd.DataFrame(split_rows).set_index("Evaluation"), disjoint)


def fit_final(processed: Processed) -> Pipeline:
    data = processed.model_data
    return new_forest().fit(data[FEATURES], data[TARGET])
