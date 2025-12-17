# time_series_analysis.py
# Handles the automated "Time-Based Analysis" feature.

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

def _find_time_and_value_cols(df):
    """Heuristically finds the most likely time and value columns."""
    time_col = None
    # First, check for actual datetime types
    for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        time_col = col
        break
    # If none, check by name
    if not time_col:
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='raise')
                    time_col = col
                    break
                except (ValueError, TypeError, pd.errors.ParserError):
                    continue

    if not time_col:
        return None, None

    # Find a suitable numeric value column (not an ID)
    value_col = None
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Avoid columns that look like IDs
        if 'id' not in col.lower() and 'key' not in col.lower() and 'number' not in col.lower() and df[col].nunique() > 1:
            value_col = col
            break
    
    if not value_col and len(numeric_cols) > 0: # Fallback to first numeric if no ideal one is found
        value_col = numeric_cols[0]

    return time_col, value_col

def perform_time_series_analysis(df, be):
    """
    Performs and displays an automated time-series analysis.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze.
        be (module): The backend module containing helper functions.
    """
    st.header("Time-Based Analysis")
    
    time_col, value_col = _find_time_and_value_cols(df)

    if not time_col or not value_col:
        st.info("Could not automatically detect suitable time and value columns for time-series analysis.")
        return

    st.markdown(f"Performing time-series analysis on value column **`{value_col}`** over time column **`{time_col}`**.")

    # Prepare data
    ts_df = df[[time_col, value_col]].copy()
    ts_df[time_col] = pd.to_datetime(ts_df[time_col])
    ts_df = ts_df.set_index(time_col).sort_index()
    
    # Resample to daily frequency for consistent analysis
    ts_daily = ts_df[value_col].resample('D').mean().fillna(method='ffill')

    if len(ts_daily) < 2:
        st.warning("Not enough data points for a meaningful time-series analysis.")
        return

    # --- 1. Overall Trend ---
    with st.expander("Overall Trend Analysis"):
        st.write(f"The plot below shows the overall trend of **`{value_col}`** over time.")
        be.plot_time_series(ts_daily.reset_index(), time_col, value_col)

    # --- 2. Seasonality Decomposition ---
    with st.expander("Seasonality and Trend Decomposition"):
        st.write("This analysis separates the time series into three components: the overall trend, seasonal patterns, and residuals (noise).")
        # The period for seasonality can be tricky. We assume weekly (7) if data spans more than 2 weeks.
        period = 7 if (ts_daily.index.max() - ts_daily.index.min()).days > 14 else None
        if period:
            try:
                decomposition = seasonal_decompose(ts_daily, model='additive', period=period)
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
                decomposition.observed.plot(ax=ax1, legend=False)
                ax1.set_ylabel('Observed')
                decomposition.trend.plot(ax=ax2, legend=False)
                ax2.set_ylabel('Trend')
                decomposition.seasonal.plot(ax=ax3, legend=False)
                ax3.set_ylabel('Seasonal')
                decomposition.resid.plot(ax=ax4, legend=False)
                ax4.set_ylabel('Residual')
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.error(f"Could not perform seasonal decomposition: {e}")
        else:
            st.info("Not enough data to determine a seasonal period for decomposition.")

    # --- 3. Monthly and Weekly Patterns ---
    with st.expander("Monthly & Weekly Patterns"):
        st.write("These plots show the average value for each month and day of the week, helping to identify peak and low periods.")
        
        # Monthly
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        monthly_avg = ts_daily.groupby(ts_daily.index.month).mean()
        monthly_avg.index = pd.to_datetime(monthly_avg.index, format='%m').strftime('%B')
        sns.barplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax[0])
        ax[0].set_title(f'Average {value_col} by Month')
        ax[0].tick_params(axis='x', rotation=45)

        # Weekly
        weekly_avg = ts_daily.groupby(ts_daily.index.day_name()).mean()
        weekly_avg = weekly_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        sns.barplot(x=weekly_avg.index, y=weekly_avg.values, ax=ax[1])
        ax[1].set_title(f'Average {value_col} by Day of Week')
        ax[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()