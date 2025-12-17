# automated_cleaning.py
# Generates a report with suggested data cleaning and preparation steps.

import streamlit as st
import pandas as pd
import numpy as np

def generate_cleaning_suggestions(df):
    """
    Analyzes the dataframe and suggests data cleaning and preparation actions.
    """
    st.subheader("Data Cleaning & Preparation Suggestions")
    st.markdown("Based on the data audit, here are recommended actions to prepare your data for analysis. These can be performed in the 'Data Preprocessing' tab.")

    # --- 1. Handle Missing Values ---
    with st.expander("💡 1. Suggestions for Missing Values"):
        missing_summary = df.isnull().sum()
        cols_with_missing = missing_summary[missing_summary > 0].index.tolist()
        if not cols_with_missing:
            st.success("No missing values to handle.")
        else:
            st.write("The following columns have missing data. Consider these strategies:")
            for col in cols_with_missing:
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.markdown(f"- **`{col}`** (Numeric): Impute with the **median** to avoid sensitivity to outliers, or with the **mean** if the data is normally distributed.")
                else:
                    st.markdown(f"- **`{col}`** (Categorical): Impute with the **mode** (most frequent value) or a new category like 'Unknown'.")

    # --- 2. Remove Duplicates ---
    with st.expander("💡 2. Suggestions for Duplicates"):
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            st.warning(f"Found **{num_duplicates}** duplicate rows. It is highly recommended to remove them to prevent skewed analysis.")
        else:
            st.success("No duplicate rows found.")

    # --- 3. Correct Data Types ---
    with st.expander("💡 3. Suggestions for Data Types"):
        st.info("Review the 'Data Type' column in the 'Automated Quick Insights' table above. If a column is misrepresented (e.g., a date stored as an object/string), use the 'Correct Data Types' tool in the 'Data Preprocessing' tab.")

    # --- 4. Standardize Categories ---
    with st.expander("💡 4. Suggestions for Categorical Standardization"):
        st.info("Review the consistency checks in the 'Data Quality Audit' report. If you find variations (e.g., 'USA' vs 'U.S.A.'), use the 'Standardize Categories' tool in the 'Data Preprocessing' tab to merge them.")

    # --- 5. Create Derived Columns (Feature Engineering) ---
    with st.expander("💡 5. Suggestions for Derived Columns"):
        st.markdown("""
        Consider creating new features to enhance your analysis. This is a creative step that depends on your goals. Examples include:
        - **Profit:** If you have `Revenue` and `Cost` columns, create `Profit` by subtracting one from the other.
        - **Time-based Features:** If you have a `Date` column, you can extract the `Month`, `Quarter`, or `Day of Week`.
        - **Binning:** If you have a numeric column like `Age`, you can group it into categorical bins like `18-25`, `26-35`, etc.
        
        These actions can be performed using the tools in the 'Data Preprocessing' tab.
        """)