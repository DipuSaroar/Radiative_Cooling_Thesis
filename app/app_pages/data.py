import pandas as pd
import streamlit as st

import charts
import pipeline as p
import state

dataset = state.active()
processed = dataset.processed
data = processed.model_data
summary = processed.summary

st.header("Data and graphs")
st.caption(f"**{dataset.name}**: {state.describe(dataset)}")

with st.container(horizontal=True):
    st.metric("Clean bins", f"{summary['retained_bins']:,}", border=True)
    st.metric("Coverage of the shared period", f"{summary['coverage_pct']:.1f}%", border=True)
    st.metric("Deleted periods", f"{summary['excluded_periods']:,}", border=True)
    st.metric("Interpolated values", "0", border=True)

# --- Time series --------------------------------------------------------------------------
st.subheader("Time series without gaps")
st.caption("Each point is a 3-minute mean after the 20-sample moving average. Deleted periods are cut out "
           "of the time axis (dotted lines). Drag the slider or scroll on a chart to zoom.")
options = ["All variables", *p.PLOT_ORDER]
choice = st.segmented_control("Variable", options, default="All variables", key="series_choice",
                              format_func=lambda c: c if c == "All variables" else p.LABELS[c][0])
if choice in (None, "All variables"):
    for row in range(0, len(p.PLOT_ORDER), 2):
        for column, name in zip(st.columns(2), p.PLOT_ORDER[row:row + 2]):
            with column.container(border=True):
                st.markdown(f"**{p.LABELS[name][0]}** ({p.LABELS[name][1]})")
                st.echarts_chart(charts.variable_chart(data, name), height=260,
                                 alt=f"{p.LABELS[name][0]} over the retained bins, without gaps")
else:
    with st.container(border=True):
        st.markdown(f"**{p.LABELS[choice][0]}** ({p.LABELS[choice][1]})")
        st.echarts_chart(charts.variable_chart(data, choice), height=420,
                         alt=f"{p.LABELS[choice][0]} over the retained bins, without gaps")
        stats = data[choice].describe()
        with st.container(horizontal=True):
            for label, key in [("Mean", "mean"), ("Minimum", "min"), ("Median", "50%"), ("Maximum", "max")]:
                st.metric(label, f"{stats[key]:.2f} {p.LABELS[choice][1]}")

# --- Before / after --------------------------------------------------------------------------
st.subheader("Effect of the moving average")
days = pd.Series(data.index.normalize().unique())
left, right = st.columns([1, 3], vertical_alignment="bottom")
with left:
    variable = st.selectbox("Variable", p.PLOT_ORDER, key="compare_variable",
                            format_func=lambda c: p.LABELS[c][0])
    start = st.selectbox("First day of a 3-day window", days, index=min(2, len(days) - 1),
                         key="compare_start", format_func=lambda d: f"{d:%d %b %Y}")
window = slice(start, start + pd.Timedelta("3D"))
compare = pd.DataFrame({"Plain 3-min mean": processed.unsmoothed.loc[window, variable],
                        "Moving average + 3-min mean": data.loc[window, variable]})
with right:
    st.echarts_chart(
        charts.gapless_lines(compare, list(compare.columns), list(compare.columns), ["#9fb7cf", "#c83d4b"],
                             p.LABELS[variable][1]),
        height=330, alt=f"{p.LABELS[variable][0]} before and after smoothing over three days")

# --- Relationships -------------------------------------------------------------------------------
left, right = st.columns(2)
with left.container(border=True):
    st.markdown("**Mean daily cycle**")
    st.echarts_chart(charts.daily_cycle(data), height=320,
                     alt="Hourly mean net heat flux and solar irradiance over the day")
with right.container(border=True):
    st.markdown("**Correlation between variables**")
    st.echarts_chart(charts.correlation_heatmap(data, p.PLOT_ORDER), height=320,
                     alt="Pearson correlation matrix of the six variables")

# --- Tables ---------------------------------------------------------------------------------------
with st.expander("Deleted periods (bins missing at least one variable)"):
    runs = processed.excluded_runs.sort_values("Deleted bins", ascending=False)
    st.dataframe(runs, hide_index=True, alt="Deleted incomplete periods, longest first",
                 column_config={"Hours": st.column_config.NumberColumn(format="%.2f")})

st.subheader("Clean dataset")
st.dataframe(data.round(3), alt="Clean modelling table: one row per retained 3-minute bin", height=300)
st.download_button("Download clean data (CSV)", data.to_csv(float_format="%.6g").encode(),
                   file_name=f"clean_{dataset.key}.csv", mime="text/csv", icon=":material/download:")
