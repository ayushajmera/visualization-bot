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

    if st.button("Run Full Automated Analysis"):
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

        st.success("Full automated analysis complete!")