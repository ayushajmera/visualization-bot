# automated_insights.py
# Generates text-based automated insights from a dataframe.

import streamlit as st
import pandas as pd

def _find_time_column(df):
    """Heuristically finds the most likely time/date column."""
    for col in df.columns:
        # Prioritize columns with 'date' or 'time' in their name
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col], errors='raise')
                return col
            except (ValueError, TypeError, pd.errors.ParserError):
                continue
    
    # If no named match, try converting other object columns
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            return col
        except (ValueError, TypeError, pd.errors.ParserError):
            continue
    return None

def _find_id_column(df):
    """Heuristically finds a potential primary key or ID column."""
    for col in df.columns:
        # Check for common ID names
        if 'id' in col.lower() or 'key' in col.lower() or 'number' in col.lower():
            # An ID column should have a high number of unique values
            if df[col].nunique() / len(df) > 0.8:
                return col
    return None

def generate_text_insights(df):
    """
    Generates a text-based summary answering common data analysis questions.
    """
    st.subheader("Automated Quick Insights")

    # --- Question 1: How many rows and columns? ---
    st.markdown("#### 1. How many rows and columns?")
    st.write(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    # --- Question 2: What does each column represent? ---
    st.markdown("#### 2. What does each column represent?")
    st.write("Below is a summary of each column's data type and basic statistics:")
    
    summary_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_vals = df[col].nunique()
        missing_vals = df[col].isnull().sum()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            description = f"Numeric. Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            description = f"Datetime. From {df[col].min()} to {df[col].max()}"
        else:
            top_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
            description = f"Categorical. Top value: '{top_val}'"
            
        summary_data.append([col, dtype, unique_vals, f"{(missing_vals/len(df)*100):.1f}%", description])
        
    summary_df = pd.DataFrame(summary_data, columns=["Column", "Data Type", "Unique Values", "Missing %", "Description"])
    st.dataframe(summary_df)

    # --- Question 3: What is the time period? ---
    st.markdown("#### 3. What is the time period?")
    time_col = _find_time_column(df)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        min_date, max_date = df[time_col].min(), df[time_col].max()
        duration = (max_date - min_date).days
        st.write(f"A potential time column, **`{time_col}`**, was found.")
        st.write(f"The data spans from **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}**, a total of **{duration} days**.")
    else:
        st.write("No clear time or date column was automatically detected.")

    # --- Question 4: What is the granularity? ---
    st.markdown("#### 4. What is the granularity? (e.g., per user, per day)")
    id_col = _find_id_column(df)
    if id_col:
        st.write(f"The column **`{id_col}`** appears to be a unique identifier, as it has **{df[id_col].nunique()}** unique values for **{len(df)}** rows.")
        if df[id_col].nunique() == len(df):
            st.write(f"This suggests the dataset's granularity is **one row per `{id_col}`**.")
        else:
            st.write(f"This suggests that each **`{id_col}`** may have multiple rows associated with it.")
    else:
        st.write("No clear unique identifier column (like 'user_id' or 'order_id') was automatically detected. The granularity might be at the level of events or transactions, but this requires manual inspection.")