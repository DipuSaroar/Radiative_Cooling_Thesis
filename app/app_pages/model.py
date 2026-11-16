import pandas as pd
import streamlit as st

import charts
import pipeline as p
import state

dataset = state.active()
data = dataset.processed.model_data

st.header("Model performance")
st.caption(f"**{dataset.name}**. Random Forest (100 trees, untuned scikit-learn defaults, seed 42) with "
           "min-max scaling fitted on the training bins only. Inputs: ambient temperature, relative humidity, "
           "wind speed, solar irradiance and sky temperature. Target: net heat flux.")

try:
    evaluation = state.evaluation(dataset.key, dataset.processed)
except p.PipelineError as error:
    st.error(str(error), title="This dataset is too small to evaluate the model")
    st.stop()

metrics = evaluation.metrics
validation = metrics.query("Partition == 'Validation'").set_index(["Evaluation", "Model"])
EXPLAIN = {
    "Random 80/20": "Bins shuffled, 20% held out. Measures how well the model reconstructs the campaign; "
                    "neighbouring bins share smoothing windows, so this score is optimistic.",
    "Chronological holdout": "Trained on the earlier 80% (minus a 24-hour purge), tested on the last 20%. "
                             "This is the honest estimate for predicting a later period.",
}

# --- Headline metrics ------------------------------------------------------------------------
for name, column in zip(evaluation.predictions, st.columns(len(evaluation.predictions), gap="medium")):
    forest = validation.loc[(name, "Random Forest")]
    baseline = validation.loc[(name, "Training-mean baseline")]
    with column.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(EXPLAIN[name])
        with st.container(horizontal=True):
            st.metric("RMSE", f"{forest['RMSE (W/m²)']:.2f} W/m²",
                      f"{forest['RMSE (W/m²)'] - baseline['RMSE (W/m²)']:+.1f} vs baseline", delta_color="inverse")
            st.metric("MAE", f"{forest['MAE (W/m²)']:.2f} W/m²",
                      f"{forest['MAE (W/m²)'] - baseline['MAE (W/m²)']:+.1f} vs baseline", delta_color="inverse")
            st.metric("R²", f"{forest['R²']:.3f}", f"{forest['R²'] - baseline['R²']:+.2f} vs baseline")
        st.echarts_chart(charts.measured_vs_predicted(evaluation.predictions[name]), height=340,
                         alt=f"{name}: predicted against measured net heat flux with the 1:1 line")
st.caption("Baseline = always predicting the training mean. Points on the dashed 1:1 line are perfect predictions.")

# --- Later-period holdout over time --------------------------------------------------------------
if "Chronological holdout" in evaluation.predictions:
    st.subheader("Later-period holdout over time")
    later = evaluation.predictions["Chronological holdout"].sort_index()
    st.echarts_chart(
        charts.gapless_lines(later, ["Measured", "Predicted"], ["Measured", "Predicted"],
                             ["#202020", "#c83d4b"], "W/m²"),
        height=360, alt="Measured and predicted heat flux across the chronological validation period")
    if evaluation.chronological_disjoint:
        st.caption("Verified: no raw reading is shared between the training and validation bins.")

left, right = st.columns(2)
with left.container(border=True):
    st.markdown("**What the model relies on** (permutation importance, later-period holdout)")
    st.echarts_chart(charts.importance_bars(evaluation.importance), height=280,
                     alt="Increase in error when each input is shuffled; larger means more important")
with right.container(border=True):
    st.markdown("**Distribution of errors**")
    st.echarts_chart(charts.residual_histogram(evaluation.predictions), height=280,
                     alt="Histogram of prediction errors for both evaluations")

with st.expander("All metrics, including training scores (overfitting check)"):
    st.dataframe(metrics.round(4), hide_index=True, alt="RMSE, MAE, R² and bias for every split and model")
    st.dataframe(evaluation.splits, alt="Training and validation sizes and validation periods")

# --- Campaign model on uploaded data --------------------------------------------------------------
if dataset.origin == "upload":
    st.subheader("Thesis model tested on this upload")
    st.caption("The Random Forest trained on the full thesis campaign predicts the uploaded bins it has never "
               "seen. This is the most direct test of how the model generalises to new measurements.")
    campaign = state.campaign()
    model = state.final_model(campaign.key, campaign.processed)
    applied = pd.DataFrame({"Measured": data[p.TARGET], "Predicted": model.predict(data[p.FEATURES])})
    scores = p.regression_metrics(applied.Measured, applied.Predicted)
    with st.container(horizontal=True):
        st.metric("RMSE", f"{scores['RMSE (W/m²)']:.2f} W/m²", border=True)
        st.metric("MAE", f"{scores['MAE (W/m²)']:.2f} W/m²", border=True)
        st.metric("R²", f"{scores['R²']:.3f}", border=True)
        st.metric("Bins", f"{len(applied):,}", border=True)
    st.echarts_chart(charts.gapless_lines(applied, ["Measured", "Predicted"], ["Measured", "Predicted"],
                                          ["#202020", "#c83d4b"], "W/m²"),
                     height=340, alt="Thesis-model predictions against the uploaded measurements")

# --- Try the model ------------------------------------------------------------------------------
st.subheader("Try the model")
st.caption("Set the five atmospheric conditions to predict the net heat flux. The model is trained on all "
           "clean bins of the selected dataset.")
model = state.final_model(dataset.key, dataset.processed)
inputs = {}
with st.container(horizontal=True):
    for feature in p.FEATURES:
        low, high, median = (float(data[feature].min()), float(data[feature].max()),
                             float(data[feature].median()))
        inputs[feature] = st.number_input(f"{p.LABELS[feature][0]} ({p.LABELS[feature][1]})",
                                          value=round(median, 1), step=1.0 if feature == "G" else 0.5,
                                          key=f"try_{feature}", width=190)
row = pd.DataFrame([inputs])
prediction = float(model.predict(row[p.FEATURES])[0])
outside = [p.LABELS[f][0].lower() for f in p.FEATURES
           if not data[f].min() <= inputs[f] <= data[f].max()]
st.metric("Predicted net heat flux", f"{prediction:.1f} W/m²", border=True, width="content")
if outside:
    st.warning("Outside the range seen in training: " + ", ".join(outside) +
               ". Random Forests cannot extrapolate, so treat this prediction with caution.")
