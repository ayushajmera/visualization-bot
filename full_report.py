# full_report.py
# Handles the generation of the "Full Insights Report".

import streamlit as st

def generate(df, col_name, be):
    """
    Generates a complete insights report for a given column,
    including statistics, anomaly detection, and visualizations.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        col_name (str): The name of the column to analyze.
        be (module): The backend module containing helper functions.
    """
    with st.spinner("Generating full insights report..."):
        be.print_header("Full Insights & Anomaly Report")
        # 1. Generate statistical insights
        be.generate_insights(df[col_name])
        # 2. Detect anomalies and get the list of outliers
        anomaly_report = be.detect_anomalies(df[col_name])
        outliers = anomaly_report.get('combined', [])
        # 3. Generate relevant plots for the single variable
        be.plot_histogram(df, col_name, outliers_to_plot=None, outliers_summary_count=len(outliers))
        be.plot_box_plot(df, col_name)
        # 4. Add the detailed outlier analysis section
        be.plot_outlier_analysis(df, col_name, outliers)