"""Radiative cooling: interactive project app.

Run from the project folder:  streamlit run app/streamlit_app.py
(or double-click run_app.bat)
"""

import streamlit as st

import state

st.set_page_config(page_title="Radiative cooling ML", page_icon=":material/ac_unit:", layout="wide")
state.init_state()

# A page that creates a dataset asks for it to become active on the next run.
if "pending_active" in st.session_state:
    st.session_state.active_dataset = st.session_state.pop("pending_active")

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True,
                help="What the project measures, how the data are cleaned, and the headline results."),
        st.Page("app_pages/data.py", title="Data and graphs", icon=":material/show_chart:",
                help="Gap-free time series of all six variables, smoothing before/after, daily cycle "
                     "and correlations for the selected dataset."),
        st.Page("app_pages/model.py", title="Model performance", icon=":material/model_training:",
                help="Random Forest accuracy on a random 80/20 split and a later-period holdout, "
                     "with feature importance and a live prediction tool."),
        st.Page("app_pages/upload.py", title="Upload new data", icon=":material/upload_file:",
                help="Upload raw instrument files or a CSV. They are cleaned with the thesis pipeline "
                     "and become selectable on every page."),
    ],
    position="top",
)

with st.sidebar:
    st.selectbox("Dataset", list(state.available()), key="active_dataset",
                 help="Every page shows the selected dataset. Uploads appear here after processing.")
    st.caption(state.describe(state.active()))
    st.divider()
    st.caption("Sample 2 RAFT campaign, INRiM Turin. Processing: 20-sample centred moving average, "
               "3-minute bins, incomplete bins deleted (no interpolation).")

page.run()
