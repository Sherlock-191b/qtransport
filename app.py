"""
qtransport.app
==============

Streamlit interface for the qtransport magnetotransport analysis tool.

This file contains **no physics logic**. It only orchestrates the workflow:

• data loading
• dataset creation
• model selection
• fitting execution
• visualization
• export of results

All physics computations are delegated to core and analysis modules.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_model import TransportDataset
from core.fitting_engine import fit_model

from core.models.two_band import TwoBandModel
from core.models.hln import HLNModel
from core.models.sdh import SdHModel

from analysis.tensor_conversion import resistivity_to_conductivity

from report.figure_style import apply_style
from report.report_generator import ReportGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="qtransport",
    page_icon="📈",
    layout="wide"
)

st.title("qtransport – Magnetotransport Analysis Tool")
st.markdown("Single-temperature magnetotransport analysis for condensed matter experiments.")


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Analysis Controls")

temperature = st.sidebar.number_input(
    "Temperature (K)",
    min_value=0.1,
    max_value=500.0,
    value=2.0
)

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Two Band Model",
        "HLN Weak Localization",
        "SdH Oscillation Model"
    ]
)


# ============================================================
# DATA INPUT SECTION
# ============================================================

st.header("1. Data Input")

input_mode = st.radio(
    "Select data input method",
    ["Upload CSV", "Manual Entry"]
)

dataset = None

# ------------------------------------------------------------
# CSV Upload
# ------------------------------------------------------------

if input_mode == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload magnetotransport CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        required_columns = ["B_field", "rho_xx", "rho_xy"]

        if all(col in df.columns for col in required_columns):

            dataset = TransportDataset(
                B_field=df["B_field"].values,
                rho_xx=df["rho_xx"].values,
                rho_xy=df["rho_xy"].values,
                temperature=temperature,
                metadata={"source": "uploaded_csv"}
            )

        else:
            st.error("CSV must contain columns: B_field, rho_xx, rho_xy")


# ------------------------------------------------------------
# Manual Data Entry
# ------------------------------------------------------------

elif input_mode == "Manual Entry":

    st.write("Enter comma-separated values.")

    B_input = st.text_area("Magnetic Field B (T)")
    rho_xx_input = st.text_area("rho_xx")
    rho_xy_input = st.text_area("rho_xy")

    if st.button("Create Dataset"):

        try:

            B = np.array([float(x) for x in B_input.split(",")])
            rho_xx = np.array([float(x) for x in rho_xx_input.split(",")])
            rho_xy = np.array([float(x) for x in rho_xy_input.split(",")])

            dataset = TransportDataset(
                B_field=B,
                rho_xx=rho_xx,
                rho_xy=rho_xy,
                temperature=temperature,
                metadata={"source": "manual_input"}
            )

            st.success("Dataset created successfully.")

        except Exception as e:
            st.error(f"Invalid input: {e}")


# ============================================================
# DATA VISUALIZATION
# ============================================================

if dataset is not None:

    st.header("2. Data Visualization")

    apply_style()

    fig, ax = plt.subplots()

    ax.plot(dataset.B_field, dataset.rho_xx, label="rho_xx")
    ax.plot(dataset.B_field, dataset.rho_xy, label="rho_xy")

    ax.set_xlabel("Magnetic Field (T)")
    ax.set_ylabel("Resistivity")

    ax.legend()

    st.pyplot(fig)


# ============================================================
# MODEL SELECTION
# ============================================================

model = None

if model_name == "Two Band Model":
    model = TwoBandModel()

elif model_name == "HLN Weak Localization":
    model = HLNModel()

elif model_name == "SdH Oscillation Model":
    model = SdHModel()


# ============================================================
# FIT EXECUTION
# ============================================================

fit_result = None

if dataset is not None:

    st.header("3. Model Fitting")

    if st.button("Run Fit"):

        with st.spinner("Running nonlinear fit..."):

            fit_result = fit_model(model, dataset)

        if fit_result.success_flag:
            st.success("Fit completed successfully.")
        else:
            st.warning("Fit did not converge.")


# ============================================================
# DISPLAY FIT RESULTS
# ============================================================

if fit_result is not None:

    st.header("4. Fit Results")

    st.write("Model:", fit_result.model_name)

    param_table = pd.DataFrame({
        "Parameter": list(fit_result.parameters),
        "Value": fit_result.parameters
    })

    st.table(param_table)

    st.write("Reduced Chi-square:", fit_result.reduced_chi_square)


# ============================================================
# FIT VISUALIZATION
# ============================================================

if fit_result is not None:

    st.header("5. Fit Visualization")

    fig, ax = plt.subplots()

    ax.plot(dataset.B_field, dataset.rho_xx, label="Data")

    ax.plot(
        dataset.B_field,
        fit_result.fitted_curve,
        label="Fit",
        linewidth=2
    )

    ax.set_xlabel("Magnetic Field (T)")
    ax.set_ylabel("rho_xx")

    ax.legend()

    st.pyplot(fig)


# ============================================================
# EXPORT RESULTS
# ============================================================

if fit_result is not None:

    st.header("6. Export Results")

    report = ReportGenerator(dataset, fit_result)

    if st.button("Export Report"):

        report.save_report("analysis_report.pdf")

        st.success("Report saved as analysis_report.pdf")
        
        
# run it by

# Bash
# streamlit run app.py