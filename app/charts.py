"""ECharts option builders.

Every time-series chart uses a *category* x axis: retained 3-minute bins are placed
side by side, so deleted (incomplete) intervals take no space and the lines have no
gaps. A faint dotted vertical line marks each place where an interval was cut out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import GRID, LABELS

CUT_STYLE = {"color": "#9a9a9a", "type": "dotted", "width": 1}
ZOOM = [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": 6}]


def _time_labels(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.strftime("%d %b %H:%M"), index=index)


def _cuts(index: pd.DatetimeIndex, labels: pd.Series) -> list[dict]:
    jumps = index.to_series().diff().gt(pd.Timedelta(GRID)).to_numpy()
    return [{"xAxis": labels.iloc[i]} for i in np.flatnonzero(jumps)]


def _cut_marks(index, labels) -> dict:
    return {"silent": True, "symbol": "none", "label": {"show": False},
            "lineStyle": CUT_STYLE, "data": _cuts(index, labels)}


def gapless_lines(frame: pd.DataFrame, columns: list[str], names: list[str], colors: list[str],
                  unit: str, *, zoom: bool = True, show_cuts: bool = True, legend: bool = True,
                  scatter: bool = False) -> dict:
    """Several series on one gap-free time axis."""
    labels = _time_labels(frame.index)
    source = frame[columns].copy()
    source.insert(0, "time", labels.to_numpy())
    series = []
    for i, (column, name, color) in enumerate(zip(columns, names, colors)):
        item = {"type": "scatter" if scatter else "line", "name": name,
                "encode": {"x": "time", "y": column}, "color": color}
        if scatter:
            item["symbolSize"] = 3
        else:
            item.update({"showSymbol": False, "lineStyle": {"width": 1.1}})
        if i == 0 and show_cuts:
            item["markLine"] = _cut_marks(frame.index, labels)
        series.append(item)
    return {
        "animation": False,  # thousands of points: draw at once instead of sweeping in
        "dataset": {"source": source},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.2f"},
        "legend": {"show": legend and len(columns) > 1, "top": 0},
        "grid": {"left": 56, "right": 20, "top": 34 if legend and len(columns) > 1 else 16,
                 "bottom": 62 if zoom else 30},
        "xAxis": {"type": "category", "boundaryGap": False, "axisLabel": {"hideOverlap": True}},
        "yAxis": {"type": "value", "name": unit, "scale": True, "nameTextStyle": {"align": "left"}},
        "dataZoom": ZOOM if zoom else [],
        "series": series,
    }


def variable_chart(frame: pd.DataFrame, column: str, *, zoom: bool = True) -> dict:
    name, unit, color = LABELS[column]
    spec = gapless_lines(frame, [column], [name], [color], unit, zoom=zoom, legend=False)
    if column == "Q_net":
        spec["series"][0]["markLine"]["data"].append({"yAxis": 0, "lineStyle": {"color": "#888", "type": "solid"}})
    return spec


def daily_cycle(data: pd.DataFrame) -> dict:
    hourly = data.groupby(data.index.hour)[["Q_net", "G"]].mean().round(2)
    hourly.index = [f"{h:02d}:00" for h in hourly.index]
    source = hourly.rename_axis("hour").reset_index()
    return {
        "dataset": {"source": source},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.1f W/m²"},
        "legend": {"top": 0},
        "grid": {"left": 56, "right": 56, "top": 40, "bottom": 30},
        "xAxis": {"type": "category", "name": "Hour"},
        "yAxis": [{"type": "value", "name": "Q_net (W/m²)"},
                  {"type": "value", "name": "G (W/m²)", "splitLine": {"show": False}}],
        "series": [
            {"type": "line", "name": "Net heat flux", "encode": {"x": "hour", "y": "Q_net"},
             "color": LABELS["Q_net"][2], "smooth": True},
            {"type": "line", "name": "Solar irradiance", "encode": {"x": "hour", "y": "G"},
             "yAxisIndex": 1, "color": LABELS["G"][2], "smooth": True, "lineStyle": {"type": "dashed"}},
        ],
    }


def correlation_heatmap(data: pd.DataFrame, columns: list[str]) -> dict:
    corr = data[columns].corr().round(2)
    cells = [[j, i, float(corr.iloc[i, j])] for i in range(len(columns)) for j in range(len(columns))]
    return {
        "tooltip": {"position": "top"},
        "grid": {"left": 60, "right": 20, "top": 10, "bottom": 70},
        "xAxis": {"type": "category", "data": columns, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": columns, "splitArea": {"show": True}, "inverse": True},
        "visualMap": {"min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center",
                      "bottom": 0, "itemHeight": 120,
                      "inRange": {"color": ["#2166ac", "#f7f7f7", "#b2182b"]}},
        "series": [{"type": "heatmap", "data": cells, "label": {"show": True}}],
    }


def measured_vs_predicted(result: pd.DataFrame) -> dict:
    low = float(min(result.Measured.min(), result.Predicted.min()))
    high = float(max(result.Measured.max(), result.Predicted.max()))
    pad = 0.04 * max(high - low, 1)
    low, high = low - pad, high + pad
    return {
        "dataset": {"source": result[["Measured", "Predicted"]].round(3)},
        "tooltip": {"trigger": "item", "valueFormatter": "%.1f W/m²"},
        "legend": {"top": 0},
        "grid": {"left": 60, "right": 20, "top": 34, "bottom": 50},
        "xAxis": {"type": "value", "name": "Measured Q_net (W/m²)", "nameLocation": "middle",
                  "nameGap": 28, "scale": True},
        "yAxis": {"type": "value", "name": "Predicted (W/m²)", "scale": True},
        "series": [
            {"type": "scatter", "name": "Validation bins", "encode": {"x": "Measured", "y": "Predicted"},
             "symbolSize": 4, "itemStyle": {"color": "#8e2548", "opacity": 0.35}},
            {"type": "line", "name": "Perfect prediction (1:1)", "data": [[low, low], [high, high]],
             "showSymbol": False, "lineStyle": {"type": "dashed", "color": "#333"}, "tooltip": {"show": False}},
        ],
    }


def importance_bars(table: pd.DataFrame) -> dict:
    ordered = table.sort_values("RMSE increase (W/m²)")
    source = ordered.assign(Label=[LABELS[f][0] for f in ordered.Feature]).round(3)
    return {
        "dataset": {"source": source[["Label", "RMSE increase (W/m²)"]]},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.2f W/m²"},
        "grid": {"left": 140, "right": 30, "top": 10, "bottom": 40},
        "xAxis": {"type": "value", "name": "RMSE increase when shuffled (W/m²)", "nameLocation": "middle",
                  "nameGap": 26},
        "yAxis": {"type": "category"},
        "series": [{"type": "bar", "encode": {"x": "RMSE increase (W/m²)", "y": "Label"},
                    "itemStyle": {"color": "#2376b7"}}],
    }


def residual_histogram(predictions: dict[str, pd.DataFrame]) -> dict:
    values = np.concatenate([p.Residual.to_numpy() for p in predictions.values()])
    low, high = np.percentile(values, [0.5, 99.5])
    edges = np.linspace(low, high, 61)
    centers = (edges[:-1] + edges[1:]) / 2
    source = pd.DataFrame({"Residual": centers.round(1)})
    for name, p in predictions.items():
        counts, _ = np.histogram(p.Residual.clip(low, high), bins=edges)
        source[name] = 100 * counts / max(len(p), 1)
    colors = ["#2376b7", "#dc8b12"]
    return {
        "dataset": {"source": source.round(3)},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.2f"},
        "legend": {"top": 0},
        "grid": {"left": 50, "right": 20, "top": 34, "bottom": 50},
        "xAxis": {"type": "category", "name": "Prediction − measurement (W/m²)", "nameLocation": "middle",
                  "nameGap": 30},
        "yAxis": {"type": "value", "name": "% of bins"},
        "series": [{"type": "bar", "name": name, "encode": {"x": "Residual", "y": name}, "barGap": "-100%",
                    "itemStyle": {"color": colors[i % 2], "opacity": 0.55}}
                   for i, name in enumerate(predictions)],
    }
