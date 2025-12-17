# data_audit.py
# Handles the "Data Quality Audit" feature.

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def _check_categorical_consistency(df, col):
    """Checks for potential inconsistencies in a categorical column."""
    series = df[col].dropna() # Drop NA to avoid errors

    # --- Robustness Fix: Only apply .str methods to actual string instances ---
    string_series = series[series.apply(lambda x: isinstance(x, str))]
    
    if not string_series.empty:
        # Check for leading/trailing whitespace
        whitespace_issues = string_series[string_series != string_series.str.strip()]
        if not whitespace_issues.empty:
            st.warning(f"**`{col}`**: Found {len(whitespace_issues)} values with leading/trailing whitespace. Example: `{whitespace_issues.iloc[0]}`. Consider using the 'Standardize Categories' tool")

        # Check for mixed case issues (e.g., "apple" and "Apple")
        lower_unique = string_series.str.lower().nunique()
        original_unique = string_series.nunique()
        if lower_unique < original_unique:
            st.warning(f"**`{col}`**: Found inconsistent casing (e.g., 'Value' and 'value'). This could cause grouping errors. The 'Standardize Categories' tool can fix this.")

    # Check for high cardinality (too many unique values)
    unique_count = series.nunique()
    if unique_count > 50:
        st.info(f"**`{col}`**: High cardinality with {unique_count} unique values. This may not be a useful categorical feature for some analyses.")
    elif unique_count == 1:
        st.info(f"**`{col}`**: Contains only a single unique value (`{series.unique()[0]}`). This column has no variance and might be a candidate for dropping.")
    elif unique_count > 15: # For columns with a moderate number of categories, show the counts
        st.info(f"**`{col}`**: Has {unique_count} unique values. Review the list below for potential inconsistencies (e.g., 'USA' vs 'U.S.A.').")
        st.dataframe(series.value_counts().to_frame("Count").head(15))
        if unique_count > 15:
            st.write(f"...and {unique_count - 15} more.")

def perform_audit(df, be):
    """
    Performs a comprehensive data quality audit on the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to audit.
        be (module): The backend module (not used here but kept for consistency).
    """
    st.header("Data Quality Audit Report")
    st.markdown("This report checks the dataset for common quality issues to ensure its trustworthiness before analysis.")

    # --- 1. Missing Values ---
    with st.expander("✅ Missing Values Analysis", expanded=True):
        st.markdown("#### Which columns have missing data?")
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if missing_summary.empty:
            st.success("No missing values found in the dataset.")
        else:
            missing_df = pd.DataFrame({
                'Missing Count': missing_summary,
                'Missing %': (missing_summary / len(df)) * 100
            })
            st.dataframe(missing_df)

            st.markdown("#### Is missing data random or systematic?")
            st.write("The heatmap below visualizes the distribution of missing values. Patterns (like vertical or horizontal bands) can indicate systematic issues.")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
            ax.set_title("Missing Values Heatmap")
            st.pyplot(fig)

    # --- 2. Duplicates ---
    with st.expander("✅ Duplicate Data Analysis", expanded=True):
        st.markdown("#### Are there duplicate rows?")
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            st.warning(f"Found **{num_duplicates}** complete duplicate rows in the dataset.")
        else:
            st.success("No complete duplicate rows found.")

        st.markdown("#### Are there duplicate IDs?")
        # Heuristically find an ID column
        id_col = None
        for col in df.columns:
            if 'id' in col.lower() or 'key' in col.lower() or 'number' in col.lower():
                if df[col].nunique() / len(df) > 0.8:
                    id_col = col
                    break
        if id_col:
            num_duplicate_ids = df[id_col].duplicated().sum()
            if num_duplicate_ids > 0:
                st.warning(f"In the potential ID column **`{id_col}`**, found **{num_duplicate_ids}** duplicate IDs.")
            else:
                st.success(f"No duplicate values found in the potential ID column **`{id_col}`**.")
        else:
            st.info("No obvious ID column was automatically detected to check for duplicate IDs.")

    # --- 3. Invalid Values ---
    with st.expander("✅ Invalid & Out-of-Range Value Analysis", expanded=True):
        st.markdown("""
        **Outlier Detection using the IQR Method**
        
        This check identifies potential outliers using the Interquartile Range (IQR) method. The formula is:
        - `IQR = Q3 (75th percentile) - Q1 (25th percentile)`
        - `Lower Bound = Q1 - 1.5 * IQR`
        - `Upper Bound = Q3 + 1.5 * IQR`
        
        Any data point outside this range is considered a potential outlier.
        """)

        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            # Check for negative values in columns that shouldn't have them
            if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'price', 'quantity', 'amount', 'count']):
                negative_values = df[df[col] < 0]
                if not negative_values.empty:
                    st.warning(f"Column **`{col}`** may contain invalid negative values. Found **{len(negative_values)}** instances.")

            # Check for out-of-range values (outliers)
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                st.warning(f"Column **`{col}`** has **{len(outliers)}** potential out-of-range values (outliers) based on the IQR method.")
        
        # Check for impossible dates
        time_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='raise')
                    time_col = col
                    break
                except Exception:
                    st.warning(f"Column **`{col}`** looks like a date but could not be parsed. It may contain invalid date formats.")
        if time_col:
            st.success(f"Successfully parsed potential date column **`{time_col}`**.")

    # --- 4. Consistency ---
    with st.expander("✅ Data Consistency Checks", expanded=True):
        st.markdown("#### Are category names consistent?")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("Checking for potential inconsistencies in categorical columns (e.g., extra spaces, slight variations).")
            for col in categorical_cols:
                st.write(f"**Column: `{col}`**")
                _check_categorical_consistency(df, col)
        else:
            st.info("No categorical columns found to check for consistency.")

        st.markdown("#### Are units consistent?")
        st.info("Unit consistency (e.g., ensuring all weights are in 'kg' or all prices are in 'USD') cannot be automated and requires manual inspection and domain knowledge.")