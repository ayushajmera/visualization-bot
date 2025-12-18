# streamlit_app.py
# UI layer for the Visualization Bot.

import sys
import os
import streamlit as st
import pandas as pd

# Add the project root to the Python path
# This ensures that 'backend' and 'full_report' can be imported by any module.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import full_report as fr
import Backend as be # Import the backend logic
import one_click_analyst as oca # Import the new feature module
import automated_insights as ai # Import the new insights module
import automated_cleaning as ac # Import the cleaning suggestions module
import exploratory_data_analysis as eda # Import the new EDA module
import time_series_analysis as tsa # Import the new time-series module
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def reset_analysis_type():
    """Callback to reset the analysis type in session state."""
    st.session_state.analysis_type = "Select an option"

# Define a fragment for the One-Click Analyst to prevent full app reruns on interaction
if hasattr(st, "fragment"):
    @st.fragment
    def run_oca_fragment(df, be):
        oca.generate_report(df, be)
elif hasattr(st, "experimental_fragment"):
    @st.experimental_fragment
    def run_oca_fragment(df, be):
        oca.generate_report(df, be)
else:
    def run_oca_fragment(df, be):
        oca.generate_report(df, be)

def main():
    """Main function to run the Visualization Bot."""
    st.set_page_config(page_title="Visualization Bot", layout="wide")
    st.title("Automated Analysis Bot")

    # Use session state to store the dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    with st.sidebar:
        st.header("1. Load Data")
        uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", type=['csv', 'xlsx', 'xls', 'json', 'txt'], key="file_uploader")
        
        if uploaded_file is not None:
            # Check if it's a new file before reloading
            if st.session_state.df is None or uploaded_file.name != st.session_state.get('file_name', ''):
                st.session_state.df = be.load_dataset(uploaded_file)
                st.session_state.processed_df = st.session_state.df.copy() if st.session_state.df is not None else None
                st.session_state.file_name = uploaded_file.name
                # Reset One-Click Analyst run flag when a new dataset is loaded so old results don't persist
                if 'oca_run' in st.session_state:
                    st.session_state.oca_run = False

        # --- Appearance / Palette Selector ---
        st.markdown("---")
        st.header("Appearance")
        try:
            palette_choice = st.selectbox("Select color palette for visuals:", options=list(be.PALETTES.keys()), index=0, key='ui_palette')
            be.set_color_palette(palette_choice)
        except Exception:
            # In case PALETTES is not available for some reason, silently continue
            pass

    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        st.sidebar.header("Workflow")

        workflow_step = st.sidebar.radio("Choose a step:",
                                         ["One-Click Data Analyst", "Data Preprocessing", "Custom Analysis"],
                                         key="workflow_step",
                                         on_change=reset_analysis_type)

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
            
            # --- Feature Scaling ---
            st.subheader("4. Feature Scaling")
            numeric_cols_for_scaling = df.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols_for_scaling:
                st.info("No numeric columns available for scaling.")
            else:
                cols_to_scale = st.multiselect("Select numeric columns to scale:", options=numeric_cols_for_scaling, key="scale_cols")
                scaler_type = st.selectbox("Select scaler type:", ["StandardScaler", "MinMaxScaler"])

                if st.button("Apply Scaling"):
                    if cols_to_scale:
                        st.session_state.processed_df = be.scale_features(df, cols_to_scale, scaler_type)
                        st.rerun()
                    else:
                        st.warning("Please select at least one column to scale.")
            
            # --- Correct Data Types ---
            st.subheader("5. Correct Data Types")
            col_to_convert = st.selectbox("Select a column to convert:", options=df.columns, key="convert_dtype_col")
            if col_to_convert:
                target_type = st.selectbox("Select target data type:", ["int", "float", "datetime", "string"], key="convert_dtype_type")
                if st.button("Convert Data Type"):
                    st.session_state.processed_df = be.correct_data_type(df, col_to_convert, target_type)
                    st.rerun()

            # --- Standardize Categories ---
            st.subheader("6. Standardize Categories")
            categorical_cols_for_std = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols_for_std:
                st.info("No categorical columns available for standardization.")
            else:
                col_to_standardize = st.selectbox("Select a categorical column to standardize:", options=categorical_cols_for_std, key="standardize_col")
                if col_to_standardize:
                    unique_values = df[col_to_standardize].unique().tolist()
                    values_to_merge = st.multiselect("Select values to merge/rename:", options=unique_values, key="standardize_vals")
                    new_value = st.text_input("Enter the new standardized category name:", key="standardize_new_val")
                    if st.button("Standardize Categories"):
                        if values_to_merge and new_value:
                            st.session_state.processed_df = be.standardize_categories(df, col_to_standardize, values_to_merge, new_value)
                            st.rerun()
                        else:
                            st.warning("Please select values to merge and provide a new category name.")
            
            # --- Create Derived Columns ---
            st.subheader("7. Create Derived Columns")
            derivation_type = st.selectbox("Select a method to create a new column:", 
                                           ["From two numeric columns (e.g., A - B)", 
                                            "Extract from a datetime column", 
                                            "Create bins from a numeric column (e.g., Age to Age Group)"])

            if derivation_type == "From two numeric columns (e.g., A - B)":
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) >= 2:
                    col_a = st.selectbox("Select the first column (A):", numeric_cols, key="derive_a")
                    operation = st.selectbox("Select operation:", ["+", "-", "*", "/"], key="derive_op")
                    col_b = st.selectbox("Select the second column (B):", numeric_cols, key="derive_b")
                    new_col_name = st.text_input("Enter new column name (e.g., Profit):", key="derive_new_name")
                    if st.button("Create Derived Column"):
                        if new_col_name:
                            st.session_state.processed_df = be.create_derived_column_numeric(df, col_a, operation, col_b, new_col_name)
                            st.rerun()
                        else:
                            st.warning("Please provide a name for the new column.")
                else:
                    st.warning("At least two numeric columns are required for this operation.")
            
            elif derivation_type == "Extract from a datetime column":
                # Heuristically find potential datetime columns
                potential_time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower()]
                if not potential_time_cols:
                    st.warning("No potential datetime columns detected. Please convert a column to datetime format first using the 'Correct Data Types' tool.")
                else:
                    time_col = st.selectbox("Select the datetime column to extract from:", potential_time_cols, key="derive_time_col")
                    parts_to_extract = st.multiselect("Select parts to extract:", 
                                                      ["Year", "Month", "Day", "Quarter", "Day of Week", "Day Name", "Week of Year"],
                                                      key="derive_time_parts")
                    if st.button("Extract Time-based Features"):
                        if time_col and parts_to_extract:
                            st.session_state.processed_df = be.create_derived_column_datetime(df, time_col, parts_to_extract)
                            st.rerun()
                        else:
                            st.warning("Please select a datetime column and at least one part to extract.")
            
            elif derivation_type == "Create bins from a numeric column (e.g., Age to Age Group)":
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns available for binning.")
                else:
                    col_to_bin = st.selectbox("Select the numeric column to bin:", numeric_cols, key="derive_bin_col")
                    num_bins = st.number_input("Enter the number of bins (e.g., 5):", min_value=2, max_value=50, value=5, key="derive_bin_num")
                    st.info("This process, also known as discretization, divides the continuous numeric data into a specified number of discrete intervals (bins). For example, binning an 'Age' column into 5 bins could create groups like '0-15', '16-30', etc. This is useful for turning numeric data into categorical data for analysis.")
                    new_col_name = st.text_input("Enter new column name (e.g., Age_Group):", key="derive_bin_name")
                    if st.button("Create Binned Column"):
                        if new_col_name and col_to_bin:
                            st.session_state.processed_df = be.create_binned_column(df, col_to_bin, num_bins, new_col_name)
                            st.rerun()
                        else:
                            st.warning("Please select a column and provide a name for the new binned column.")

        elif workflow_step == "Custom Analysis":
            st.sidebar.header("3. Choose Analysis Type")
            analysis_type = st.sidebar.radio("Select Analysis",
                                             ["Select an option", "Full Insights Report", "Univariate Analysis", "Multivariate Analysis", "Temporal Analysis"],
                                             key="analysis_type",
                                             index=0) # Default to "Select an option"

            if analysis_type == "Full Insights Report":
                be.print_header("Full Insights & Anomaly Report")
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found for insights report.")
                else:
                    col_name = st.selectbox("Select a numeric column for a detailed report:", numeric_cols)
                    if col_name:
                        if st.button("Generate Report"):
                            fr.generate(df, col_name, be) # Pass the backend module

            elif analysis_type == "Univariate Analysis":
                be.print_header("Univariate Analysis (Single Variable)")
                if st.button("Run Full Univariate Analysis"):
                    with st.spinner("Running full univariate analysis... This might take a moment."):
                        st.subheader("Analysis of Numeric Columns")
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        for col in numeric_cols:
                            with st.expander(f"Analysis for '{col}'"):
                                hist_fig, hist_buf = be.plot_histogram(df, col)
                                st.pyplot(hist_fig)
                                st.download_button("Download Histogram", hist_buf, f"histogram_{col}.png", "image/png")

                                box_fig, box_buf = be.plot_box_plot(df, col)
                                st.pyplot(box_fig)
                                st.download_button("Download Box Plot", box_buf, f"boxplot_{col}.png", "image/png")

                        st.subheader("Analysis of Categorical Columns")
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                        for col in categorical_cols:
                            with st.expander(f"Analysis for '{col}'"):
                                bar_fig, bar_buf = be.plot_bar_chart(df, col)
                                st.pyplot(bar_fig)
                                st.download_button("Download Bar Chart", bar_buf, f"barchart_{col}.png", "image/png")

                                pie_fig, pie_buf = be.plot_pie_chart(df, col)
                                st.pyplot(pie_fig)
                                st.download_button("Download Pie Chart", pie_buf, f"piechart_{col}.png", "image/png")

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
                            scatter_fig = be.plot_scatter_plot(df, col1, col2)
                            if scatter_fig:
                                st.plotly_chart(
                                    scatter_fig,
                                    width='stretch',
                                    key=f"custom_scatter_{str(col1).replace(' ','_').replace('.','_')}__{str(col2).replace(' ','_').replace('.','_')}"
                                )
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

        elif workflow_step == "One-Click Data Analyst":
            run_oca_fragment(df, be)

    else:
        st.info("Awaiting for a dataset to be uploaded.")

if __name__ == "__main__":
    main()