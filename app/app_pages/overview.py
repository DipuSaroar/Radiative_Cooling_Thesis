import streamlit as st

import charts
import pipeline as p
import state

dataset = state.active()
summary = dataset.processed.summary

st.title("Predicting net heat flux of a passive radiative cooling material")
st.markdown(
    "A radiative cooling sample (Sample 2, RAFT set-up at INRiM, Turin) was measured outdoors in spring 2026. "
    "A heat-flux sensor recorded the **net heat flux** through the sample, while a weather station and an "
    "infrared sky thermometer recorded the conditions around it. This project cleans the three instruments' "
    "logs into one gap-free table and trains a **Random Forest** to predict the heat flux from five "
    "atmospheric inputs."
)
st.caption(f"Showing: **{dataset.name}**. Change the dataset in the sidebar.")

evaluation = state.evaluation(dataset.key, dataset.processed)
validation = evaluation.metrics.query("Partition == 'Validation' and Model == 'Random Forest'").set_index("Evaluation")

with st.container(horizontal=True):
    st.metric("Clean 3-minute bins", f"{summary['retained_bins']:,}", border=True)
    st.metric("Recording period", f"{(summary['last'] - summary['first']).days} days", border=True)
    st.metric("Incomplete bins deleted", f"{summary['excluded_bins']:,}", border=True)
    for name in validation.index:
        st.metric(f"R², {name.lower()}", f"{validation.loc[name, 'R²']:.3f}", border=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("From raw logs to a model")
    st.mermaid_chart(
        """
flowchart LR
    A[RAFT heat-flux logger<br/>Q_net] --> D
    B[Weather station<br/>T_amb, RH, v, G] --> D
    C[IR sky thermometer<br/>T_sky] --> D
    D[Parse and merge raw files<br/>decimal commas, duplicates] --> E[20-sample centred<br/>moving average]
    E --> F[3-minute bin means]
    F --> G[Delete bins missing<br/>any variable]
    G --> H[Random Forest<br/>5 inputs to Q_net]
""",
        alt="Pipeline: three instruments are merged, smoothed, binned, incomplete bins deleted, then modelled",
    )
    st.markdown(
        "- **Nothing is invented.** A 3-minute bin is kept only when all six variables were recorded in it. "
        "Missing periods are deleted, never interpolated, so every graph shows recorded data only.\n"
        "- **Gap-free graphs.** Retained bins are placed side by side; a faint dotted line marks each place "
        "where a deleted period was cut out.\n"
        "- **Two honest evaluations.** A shuffled 80/20 split measures reconstruction within the campaign; a "
        "chronological holdout (the last 20%, with a 24-hour purge) measures prediction of a later period."
    )
with right:
    st.subheader("Model variables")
    st.dataframe(
        {"Variable": p.PLOT_ORDER,
         "Meaning": [p.LABELS[c][0] for c in p.PLOT_ORDER],
         "Unit": [p.LABELS[c][1] for c in p.PLOT_ORDER],
         "Role": ["Target" if c == p.TARGET else "Input" for c in p.PLOT_ORDER]},
        hide_index=True, alt="The five model inputs and the heat-flux target with units",
    )
    st.subheader("Net heat flux over the whole period")
    st.echarts_chart(charts.variable_chart(dataset.processed.model_data, "Q_net", zoom=False), height=240,
                     alt="Net heat flux for every retained bin, shown without gaps")

st.subheader("Explore")
with st.container(horizontal=True):
    st.page_link("app_pages/data.py", label="Data and graphs", icon=":material/show_chart:")
    st.page_link("app_pages/model.py", label="Model performance", icon=":material/model_training:")
    st.page_link("app_pages/upload.py", label="Upload new data", icon=":material/upload_file:")
