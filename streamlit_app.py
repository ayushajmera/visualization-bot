# streamlit_app.py
# UI layer for the Visualization Bot.

import streamlit as st
import pandas as pd
import numpy as np
import Backend as be # Import the backend logic
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def main():
    """Main function to run the Visualization Bot."""
    st.set_page_config(page_title="Visualization Bot", layout="wide")
    st.title("📊 Automated Analysis Bot")

    # Use session state to store the dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    with st.sidebar:
        st.header("1. Load Data")
        uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", type=['csv', 'xlsx', 'xls', 'json', 'txt'])
        
        if uploaded_file is not None:
            # Check if it's a new file before reloading
            if st.session_state.df is None or uploaded_file.name != st.session_state.get('file_name', ''):
                st.session_state.df = be.load_dataset(uploaded_file)
                st.session_state.processed_df = st.session_state.df.copy() if st.session_state.df is not None else None
                st.session_state.file_name = uploaded_file.name

    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        st.sidebar.header("Workflow")
        
        workflow_step = st.sidebar.radio("Choose a step:", 
                                         ["Data Preprocessing", "Data Analysis"])

        if workflow_step == "Data Preprocessing":
            be.print_header("Data Preprocessing")
            
            st.subheader("Current Data Preview")
            st.dataframe(df.head())

            # --- Column Dropping ---
            st.subheader("1. Drop Unnecessary Columns")
            all_columns = df.columns.tolist()
            cols_to_drop = st.multiselect("Select columns to drop:", options=all_columns, key="drop_cols")
            if st.button("Drop Selected Columns"):
                if cols_to_drop:
                    st.session_state.processed_df = be.drop_columns(df, cols_to_drop)
                    st.rerun()
                else:
                    st.warning("Please select at least one column to drop.")

            # --- Missing Value Handling ---
            st.subheader("2. Handle Missing Values")
            missing_values = df.isnull().sum()
            cols_with_missing = missing_values[missing_values > 0].index.tolist()

            if not cols_with_missing:
                st.success("No missing values found in the dataset.")
            else:
                st.write("Columns with missing values:")
                st.dataframe(missing_values[missing_values > 0].astype(str).to_frame('Missing Count'))

                col_to_process = st.selectbox("Select a column to process:", cols_with_missing)
                
                # Determine column type for strategy options
                is_numeric = pd.api.types.is_numeric_dtype(df[col_to_process])
                
                if is_numeric:
                    strategies = ["Select a strategy", "Impute with Mean", "Impute with Median", "Impute with Mode", "Fill with a specific value", "Drop rows with missing values"]
                else: # Categorical
                    strategies = ["Select a strategy", "Impute with Mode", "Fill with a specific value", "Drop rows with missing values"]

                strategy = st.selectbox("Select a strategy for this column:", strategies)

                fill_value = None
                if strategy == "Fill with a specific value":
                    fill_value = st.text_input("Enter the value to fill missing entries with:")

                if st.button("Apply Strategy"):
                    if strategy != "Select a strategy":
                        st.session_state.processed_df = be.handle_missing_values(df, col_to_process, strategy, fill_value)
                        st.rerun()
                    else:
                        st.warning("Please select a valid strategy.")

            # --- Duplicate Handling ---
            st.subheader("3. Handle Duplicate Rows")
            num_duplicates = df.duplicated().sum()
            st.write(f"Number of duplicate rows found: **{num_duplicates}**")
            if num_duplicates > 0:
                with st.expander("Click to see duplicate rows"):
                    st.dataframe(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()))

            if st.button("Remove Duplicates"):
                st.session_state.processed_df = be.remove_duplicates(df)
                st.rerun()

        elif workflow_step == "Data Analysis":
            st.sidebar.header("3. Choose Analysis Type")
            analysis_type = st.sidebar.radio("Select Analysis", 
                                             ["Select an option", "Full Insights Report", "Univariate Analysis", "Multivariate Analysis", "Temporal Analysis"])

            if analysis_type == "Full Insights Report":
                be.print_header("Full Insights & Anomaly Report")
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found for insights report.")
                else:
                    col_name = st.selectbox("Select a numeric column for a detailed report:", numeric_cols)
                    if col_name:
                        if st.button("Generate Report"):
                            with st.spinner("Generating full insights report..."):
                                be.generate_insights(df[col_name])
                                anomaly_report = be.detect_anomalies(df[col_name])
                                outliers = anomaly_report.get('combined', [])
                                be.plot_histogram(df, col_name, outliers_to_plot=None, outliers_summary_count=len(outliers)) # Don't plot individual outliers here
                                be.plot_box_plot(df, col_name)
                                # Add the new detailed outlier analysis section
                                be.plot_outlier_analysis(df, col_name, outliers)

            elif analysis_type == "Univariate Analysis":
                be.print_header("Univariate Analysis (Single Variable)")
                if st.button("🚀 Run Full Univariate Analysis"):
                    with st.spinner("Running full univariate analysis... This might take a moment."):
                        st.subheader("Analysis of Numeric Columns")
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        for col in numeric_cols:
                            with st.expander(f"Analysis for '{col}'"):
                                be.plot_histogram(df, col) # Here, outliers_to_plot is None by default, which is fine.
                                be.plot_box_plot(df, col)

                        st.subheader("Analysis of Categorical Columns")
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                        for col in categorical_cols:
                            with st.expander(f"Analysis for '{col}'"):
                                be.plot_bar_chart(df, col)
                                be.plot_pie_chart(df, col)

            elif analysis_type == "Multivariate Analysis":
                be.print_header("Multivariate Analysis (Multiple Variables)")
                st.subheader("Correlation Heatmap")
                if st.button("Generate Heatmap"):
                    with st.spinner("Generating heatmap..."):
                        be.plot_correlation_heatmap(df)

                st.subheader("Pair Plot")
                if st.button("Generate Pair Plot (up to 5 columns)"):
                    with st.spinner("Generating pair plot... This can be slow for many columns."):
                        be.plot_pair_plot(df)

                st.subheader("Custom Scatter Plot")
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1 = st.selectbox("Select X-axis column:", numeric_cols, key='scatter_x')
                    col2 = st.selectbox("Select Y-axis column:", numeric_cols, key='scatter_y', index=min(1, len(numeric_cols)-1))
                    if st.button("Generate Scatter Plot"):
                        with st.spinner("Generating scatter plot..."):
                            be.plot_scatter_plot(df, col1, col2)
                else:
                    st.warning("At least two numeric columns are required for a scatter plot.")

                st.subheader("Custom 3D Scatter Plot")
                if len(numeric_cols) >= 3:
                    x_3d = st.selectbox("Select X-axis column:", numeric_cols, key='3d_x')
                    y_3d = st.selectbox("Select Y-axis column:", numeric_cols, key='3d_y', index=min(1, len(numeric_cols)-1))
                    z_3d = st.selectbox("Select Z-axis column:", numeric_cols, key='3d_z', index=min(2, len(numeric_cols)-1))
                    
                    # Optional color dimension
                    all_cols_for_color = ["None"] + df.columns.tolist()
                    color_3d = st.selectbox("Select column for color (optional):", all_cols_for_color, key='3d_color')

                    if st.button("Generate 3D Scatter Plot"):
                        with st.spinner("Generating 3D scatter plot..."):
                            be.plot_3d_scatter_plot(df, x_3d, y_3d, z_3d, color_3d)
                else:
                    st.warning("At least three numeric columns are required for a 3D scatter plot.")

            elif analysis_type == "Temporal Analysis":
                be.print_header("Temporal Analysis (Time Series)")
                df_temp = df.copy()
                potential_time_cols = []
                for col in df_temp.columns:
                    try:
                        pd.to_datetime(df_temp[col], errors='raise')
                        potential_time_cols.append(col)
                    except (ValueError, TypeError, pd.errors.ParserError):
                        continue
                
                if not potential_time_cols:
                    st.warning("No columns could be automatically converted to a datetime format for temporal analysis.")
                else:
                    time_col = st.selectbox("Select the time/date column:", potential_time_cols)
                    df_temp[time_col] = pd.to_datetime(df_temp[time_col])
                    numeric_cols = df_temp.select_dtypes(include=np.number).columns.tolist()
                    value_col = st.selectbox("Select the value column to plot over time:", numeric_cols)
                    if st.button("Generate Time Series Plot"):
                        with st.spinner("Generating time series plot..."):
                            be.plot_time_series(df_temp, time_col, value_col)

    else:
        st.info("Awaiting for a dataset to be uploaded.")

if __name__ == "__main__":
    main()