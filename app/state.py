"""Cached data access shared by all pages.

The campaign data are processed once from ``raw_data/`` and cached for every visitor.
Uploaded datasets live only in the browser session, so nothing on disk is changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import pipeline as p

CAMPAIGN = "Thesis campaign (raw_data)"


@dataclass
class Dataset:
    name: str
    key: str  # content hash, used as the cache key for the model
    processed: p.Processed
    report: p.ReadReport
    origin: str  # "campaign", "combined" or "upload"


@st.cache_data(show_spinner="Cleaning the campaign raw files…")
def _campaign_parts():
    native, report = p.read_native(p.bundled_raw_files())
    return native, report, p.process(native)


def campaign_native() -> dict[str, pd.Series]:
    return _campaign_parts()[0]


def campaign() -> Dataset:
    native, report, processed = _campaign_parts()
    return Dataset(CAMPAIGN, "campaign", processed, report, "campaign")


@st.cache_data(show_spinner="Training the Random Forest and scoring both validation splits…", max_entries=8)
def evaluation(key: str, _processed: p.Processed) -> p.Evaluation:
    return p.evaluate(_processed)


@st.cache_resource(show_spinner="Fitting the model on all retained bins…", max_entries=8)
def final_model(key: str, _processed: p.Processed):
    return p.fit_final(_processed)


def build_upload(files: list[tuple[str, bytes]], combine: bool, name: str) -> Dataset:
    digest = hashlib.sha256()
    for file_name, data in sorted(files):
        digest.update(file_name.encode())
        digest.update(data)
    native, report = p.read_native(files)
    if combine:
        native = p.combine_native(campaign_native(), native)
    processed = p.process(native)
    key = f"{'combined' if combine else 'upload'}-{digest.hexdigest()[:16]}"
    return Dataset(name, key, processed, report, "combined" if combine else "upload")


def init_state() -> None:
    if "uploads" not in st.session_state:
        st.session_state.uploads = {}  # name -> Dataset


def available() -> dict[str, Dataset]:
    return {CAMPAIGN: campaign(), **st.session_state.uploads}


def active() -> Dataset:
    datasets = available()
    return datasets.get(st.session_state.get("active_dataset", CAMPAIGN), datasets[CAMPAIGN])


def describe(dataset: Dataset) -> str:
    s = dataset.processed.summary
    return (f"{s['retained_bins']:,} clean 3-minute bins from {s['first']:%d %b %Y %H:%M} "
            f"to {s['last']:%d %b %Y %H:%M}; {s['excluded_bins']:,} incomplete bins deleted, none interpolated.")
