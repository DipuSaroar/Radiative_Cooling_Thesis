import pandas as pd
import streamlit as st

import pipeline as p
import state

st.header("Upload new data")
st.markdown(
    "Upload new measurements and they are cleaned with exactly the same pipeline as the thesis: parsed, "
    "smoothed with the 20-sample moving average, averaged into 3-minute bins, and **every bin that misses a "
    "variable is deleted**, so the graphs never show gaps or invented values. The result becomes selectable "
    "in the sidebar on every page. Nothing on disk is changed."
)

with st.expander("Which files can I upload?", icon=":material/help:"):
    st.markdown(
        "**Raw instrument logs** (any mix, any number of days). Files are recognised by their content:\n"
        "- RAFT heat-flux logger: `YYYYMMDD_RAFT.txt`\n"
        "- Weather station: `YYYYMMDD_meteo.log`\n"
        "- IR sky thermometer: `*-IR*.dat` (detailed) or `*-IR-new.txt` (simple)\n\n"
        "**Or a CSV table** with a `timestamp` column and any of the columns "
        "`Q_net, T_amb, RH, v, G, T_sky` (units as on the Overview page). Each row is treated as a raw "
        "reading and goes through the same smoothing and binning."
    )
    template = pd.DataFrame({"timestamp": ["2026-06-01 12:00:00", "2026-06-01 12:01:00"],
                             "Q_net": [-35.2, -34.8], "T_amb": [24.1, 24.2], "RH": [48, 48],
                             "v": [1.3, 1.6], "G": [812, 815], "T_sky": [-12.4, -12.5]})
    st.download_button("Download CSV template", template.to_csv(index=False).encode(),
                       file_name="upload_template.csv", mime="text/csv", icon=":material/download:")

with st.form("upload_form", border=True):
    files = st.file_uploader("Data files", type=["txt", "log", "dat", "csv"], accept_multiple_files=True,
                             key="upload_files")
    left, right = st.columns(2, vertical_alignment="bottom")
    with left:
        mode = st.segmented_control(
            "How to use the upload", ["Analyse on its own", "Add to the thesis campaign"],
            default="Analyse on its own", key="upload_mode",
            help="On its own: only the uploaded files are used, and the thesis model is also tested on them. "
                 "Added: the uploaded readings extend the campaign data and the model is retrained.")
    with right:
        name = st.text_input("Dataset name", value=f"Upload {pd.Timestamp.now():%d %b %H:%M}",
                             key="upload_name")
    submitted = st.form_submit_button("Clean and process", type="primary", icon=":material/cleaning_services:")

if submitted:
    if not files:
        st.warning("Choose at least one file first.")
    else:
        name = name.strip() or "Upload"
        if name == state.CAMPAIGN or name in st.session_state.uploads:
            name = f"{name} ({len(st.session_state.uploads) + 1})"
        try:
            with st.spinner("Reading, cleaning and binning the files…"):
                dataset = state.build_upload([(f.name, f.getvalue()) for f in files],
                                             combine=mode == "Add to the thesis campaign", name=name)
        except p.PipelineError as error:
            st.error(str(error), title="These files cannot be turned into a clean dataset")
        except Exception as error:  # unexpected file content
            st.error(f"{type(error).__name__}: {error}", title="Could not read the files")
        else:
            st.session_state.uploads[name] = dataset
            st.session_state.pending_active = name
            st.session_state.last_upload = name
            st.rerun()

# --- Result of the latest upload ----------------------------------------------------------------
latest = st.session_state.uploads.get(st.session_state.get("last_upload", ""))
if latest is not None:
    summary = latest.processed.summary
    st.success(f"**{latest.name}** is ready and selected in the sidebar. {state.describe(latest)}",
               title="Data cleaned")
    with st.container(horizontal=True):
        st.metric("Clean bins", f"{summary['retained_bins']:,}", border=True)
        st.metric("Deleted incomplete bins", f"{summary['excluded_bins']:,}", border=True)
        st.metric("Deleted periods", f"{summary['excluded_periods']:,}", border=True)
        st.metric("Coverage", f"{summary['coverage_pct']:.1f}%", border=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Files read**")
        st.dataframe(latest.report.files, hide_index=True, alt="Each uploaded file and how it was recognised")
    with right:
        st.markdown("**Readings per variable (before cleaning)**")
        st.dataframe(latest.report.readings, alt="Number and time range of readings for each variable")
    with st.container(horizontal=True):
        st.page_link("app_pages/data.py", label="See the graphs", icon=":material/show_chart:")
        st.page_link("app_pages/model.py", label="See model performance", icon=":material/model_training:")
        st.download_button("Download clean data (CSV)",
                           latest.processed.model_data.to_csv(float_format="%.6g").encode(),
                           file_name=f"clean_{latest.key}.csv", mime="text/csv", icon=":material/download:")

if st.session_state.uploads:
    st.subheader("Datasets in this session")
    st.dataframe(pd.DataFrame([{"Name": d.name, "Use": "added to campaign" if d.origin == "combined" else "on its own",
                                "Clean bins": d.processed.summary["retained_bins"],
                                "From": d.processed.summary["first"], "To": d.processed.summary["last"]}
                               for d in st.session_state.uploads.values()]),
                 hide_index=True, alt="Uploaded datasets available in the sidebar")
    if st.button("Remove all uploads", icon=":material/delete:"):
        st.session_state.uploads = {}
        st.session_state.pending_active = state.CAMPAIGN
        st.session_state.pop("last_upload", None)
        st.rerun()
