# one_click_analyst.py
# Handles the generation of the "One-Click Data Analyst" report.

import streamlit as st
import numpy as np
import automated_insights as ai # Import the new insights module
import data_audit as da # Import the audit module
import automated_cleaning as ac # Import the cleaning suggestions module
import exploratory_data_analysis as eda # Import the EDA module
import time_series_analysis as tsa # Import the new time-series module

def generate_report(df, be):
    """
    Generates a full automated report for the entire dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        be (module): The backend module containing helper functions.
    """
    be.print_header("One-Click Data Analyst")
    st.info("Click the button below to generate a full automated report for the entire dataset. This may take a few moments depending on the size of your data.")

    # Use session state to remember that the automated analysis was requested
    # This prevents the report from disappearing on subsequent Streamlit reruns (e.g., when sliders change)
    if 'oca_run' not in st.session_state:
        st.session_state.oca_run = False

    left, middle, right = st.columns([1, 2, 1])
    with left:
        if st.button("Run Full Automated Analysis"):
            st.session_state.oca_run = True

    with right:
        if st.button("Reset Full Automated Analysis"):
            st.session_state.oca_run = False

    # If the run flag is set, (re-)execute the analysis steps so the UI is rebuilt on every rerun
    if st.session_state.oca_run:
        with st.spinner("Running full automated analysis... this may take a few moments"):
            # 1. Perform Data Quality Audit first
            da.perform_audit(df, be)
            st.markdown("---") # Add a separator

            # 2. Generate text-based insights
            ai.generate_text_insights(df)
            st.markdown("---") # Add a separator

            # 3. Generate cleaning and preparation suggestions
            ac.generate_cleaning_suggestions(df)
            st.markdown("---") # Add a separator

            # 4. Perform Exploratory Data Analysis
            eda.perform_eda(df, be)
            st.markdown("---") # Add a separator

            # 5. Perform Time-Based Analysis
            tsa.perform_time_series_analysis(df, be)
            st.markdown("---") # Add a separator

        st.success("Full automated analysis complete! (Use the 'Reset' button to clear results.)")