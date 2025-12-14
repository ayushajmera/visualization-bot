# backend.py
# Core logic for data processing, analysis, and visualization.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import io
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def print_header(title: str):
    """Prints a formatted header to the console."""
    st.header(title)

def load_dataset(uploaded_file) -> pd.DataFrame | None:
    """
    Loads a dataset from a given file path (CSV, Excel, JSON, TXT).
    Automatically detects the file format and handles loading errors.
    """
    if uploaded_file is None:
        return None

    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['csv', 'txt']:
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"❌ Error: Unsupported file format '{file_extension}'. Please use CSV, Excel, or JSON.")
            return None

        st.success("✅ Dataset loaded successfully!")
        st.write("--- Dataset Shape ---")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.write("--- Columns and Data Types ---")
        st.dataframe(df.dtypes.astype(str).to_frame('Data Type'))
        
        return df

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

def generate_insights(series: pd.Series):
    """Generates and prints a full statistical analysis report for a numeric column."""
    
    # Basic Stats
    st.metric("Mean", f"{series.mean():.4f}")
    st.metric("Median", f"{series.median():.4f}")
    st.metric("Standard Deviation", f"{series.std():.4f}")
    st.metric("Minimum", f"{series.min():.4f}")
    st.metric("Maximum", f"{series.max():.4f}")
    st.metric("Missing Values", f"{series.isnull().sum()}")

    # Distribution Shape
    st.subheader("Distribution Shape")
    skewness = series.skew()
    kurtosis = series.kurtosis()
    st.metric("Skewness", f"{skewness:.4f}")
    st.metric("Kurtosis", f"{kurtosis:.4f}")
    
    # Interpretation of Skewness
    if skewness > 0.5:
        skew_interp = "Right-skewed (positively skewed). The tail is on the right."
    elif skewness < -0.5:
        skew_interp = "Left-skewed (negatively skewed). The tail is on the left."
    else:
        skew_interp = "Fairly symmetrical."
    st.write(f"**Skewness Interpretation:** {skew_interp}")

    # Normality Test
    st.subheader("Normality Test (D'Agostino's K^2)")
    if series.notna().sum() > 8: # Test requires at least 8 samples
        stat, p_value = stats.normaltest(series.dropna())
        st.metric("P-value", f"{p_value:.4f}")
        if p_value < 0.05:
            st.warning("The data likely does NOT come from a normal distribution (p < 0.05).")
        else:
            st.success("The data appears to be normally distributed (p >= 0.05).")
    else:
        st.info("Not enough data to perform normality test.")

def detect_anomalies(series: pd.Series) -> dict:
    """
    Detects anomalies using IQR, Z-Score, and Isolation Forest methods.
    Returns a dictionary with outliers from each method.
    """
    data = series.dropna()
    outliers = {}

    # 1. IQR Method
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers['iqr'] = iqr_outliers.tolist()
    st.write(f"**IQR Method:** Found {len(outliers['iqr'])} outliers.")

    # 2. Z-Score Method
    z_scores = np.abs(stats.zscore(data))
    z_score_outliers = data[z_scores > 3]
    outliers['z_score'] = z_score_outliers.tolist()
    st.write(f"**Z-Score (threshold=3) Method:** Found {len(outliers['z_score'])} outliers.")

    # 3. Isolation Forest Method
    if len(data) > 1:
        iso_forest = IsolationForest(contamination='auto', random_state=42)
        predictions = iso_forest.fit_predict(data.values.reshape(-1, 1))
        iso_forest_outliers = data[predictions == -1]
        outliers['isolation_forest'] = iso_forest_outliers.tolist()
        st.write(f"**Isolation Forest Method:** Found {len(outliers['isolation_forest'])} outliers.")
    else:
        outliers['isolation_forest'] = []
        st.write("Isolation Forest requires more than one data point.")

    # Combined Unique Outliers
    all_outliers = set(outliers['iqr']) | set(outliers['z_score']) | set(outliers['isolation_forest'])
    outliers['combined'] = sorted(list(all_outliers))
    st.subheader(f"Combined Unique Outliers: {len(outliers['combined'])}")
    if outliers['combined']:
        st.write(outliers['combined'])
    
    return outliers

def save_and_show_plot(key_suffix: str):
    """Displays the current plot and provides a download button without saving to disk."""
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    try:
        # Add a download button for the image
        st.download_button(
            label="Download Plot",
            data=buf,
            file_name=f"output_{key_suffix}.png",
            mime="image/png",
            key=f"download_{key_suffix}"
        )
        plt.clf() # Clear the figure for the next plot
    except Exception as e:
        st.error(f"❌ Error creating download button: {e}")

def plot_histogram(df: pd.DataFrame, column: str, outliers_to_plot: list | None = None, outliers_summary_count: int = 0):
    """Generates and saves a histogram, highlighting outliers if provided."""
    print_header(f"Generating Histogram for '{column}'")
    plt.figure(figsize=(12, 7))
    sns.histplot(df[column], kde=True, bins=30)
    
    if outliers_to_plot: # Only plot individual lines if a list of outliers is explicitly provided
        for outlier in outliers_to_plot:
            plt.axvline(x=outlier, color='r', linestyle='--', label=f'Outlier: {outlier:.2f}')
        # Create a legend with unique labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    plt.title(f'Histogram of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    # Add description and conclusion
    skewness = df[column].skew()
    if skewness > 0.5:
        skew_interp = "Right-skewed (positively skewed). The tail is on the right."
    elif skewness < -0.5:
        skew_interp = "Left-skewed (negatively skewed). The tail is on the left."
    else:
        skew_interp = "Fairly symmetrical."
    st.write(f"**Skewness Interpretation:** {skew_interp}")

    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This histogram shows the frequency distribution of the **`{column}`** variable.
    - {skew_interp}
    - The Kernel Density Estimate (KDE) line provides a smooth estimate of the data's distribution.
    """)
    if outliers_summary_count > 0:
        st.markdown(f"""
    - **{outliers_summary_count} potential outliers** were detected. Please refer to the accompanying box plot for their visual representation.
    """)
    save_and_show_plot(key_suffix=f"hist_{column}")

def plot_bar_chart(df: pd.DataFrame, column: str): 
    """Generates and saves a bar chart for a categorical column."""
    print_header(f"Generating Bar Chart for '{column}'")
    plt.figure(figsize=(12, 7))
    
    # Use value_counts and plot for better control, especially with many categories
    counts = df[column].value_counts().nlargest(20) # Limit to top 20 for readability
    sns.barplot(x=counts.index, y=counts.values)

    plt.title(f'Bar Chart of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    
    # Add description and conclusion
    most_frequent = counts.index[0]
    least_frequent = counts.index[-1]
    
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This bar chart displays the frequency of each category in the **`{column}`** variable (showing the top {len(counts)} categories).
    - The most frequent category is **'{most_frequent}'** with {counts.iloc[0]} occurrences.
    - The least frequent category shown is **'{least_frequent}'** with {counts.iloc[-1]} occurrences.
    - This helps in understanding the prevalence of different categories in your dataset.
    """)
    save_and_show_plot(key_suffix=f"bar_{column}")

def plot_pie_chart(df: pd.DataFrame, column: str):
    """Generates and saves a pie chart for a categorical column."""
    print_header(f"Generating Pie Chart for '{column}'")
    plt.figure(figsize=(10, 10))
    
    counts = df[column].value_counts()
    # Group small slices into 'Other' for clarity
    if len(counts) > 10:
        threshold = counts.nlargest(9).min()
        other = counts[counts < threshold].sum()
        counts = counts[counts >= threshold]
        if other > 0:
            counts['Other'] = other

    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'white'})
    plt.title(f'Pie Chart of {column}', fontsize=16)
    plt.ylabel('') # Hide the y-label which is often the column name
    
    # Add description and conclusion
    largest_slice = counts.index[0]
    largest_percentage = (counts.iloc[0] / counts.sum()) * 100
    
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This pie chart illustrates the proportional distribution of categories in the **`{column}`** variable.
    - The largest slice belongs to **'{largest_slice}'**, accounting for **{largest_percentage:.1f}%** of the total.
    - If an 'Other' slice is present, it groups together smaller categories for better readability.
    - This is useful for seeing the relative share of each category at a glance.
    """)
    save_and_show_plot(key_suffix=f"pie_{column}")

def plot_box_plot(df: pd.DataFrame, column: str):
    """Generates and saves a box plot for a numeric column."""
    print_header(f"Generating Box Plot for '{column}'")
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.grid(axis='x', alpha=0.5)
    
    # Add description and conclusion
    q1 = df[column].quantile(0.25)
    median = df[column].median()
    q3 = df[column].quantile(0.75)
    
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This box plot summarizes the distribution of the **`{column}`** variable.
    - The box represents the Interquartile Range (IQR), with the left edge at the 25th percentile (Q1: {q1:.2f}) and the right edge at the 75th percentile (Q3: {q3:.2f}).
    - The vertical line inside the box is the **median** (50th percentile): **{median:.2f}**.
    - The 'whiskers' extend to show the range of the data, typically 1.5 times the IQR. Points outside the whiskers are often considered potential outliers.
    """)
    save_and_show_plot(key_suffix=f"box_{column}")

def plot_correlation_heatmap(df: pd.DataFrame):
    """Calculates and plots the correlation heatmap for numeric columns."""
    print_header("Generating Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns (at least 2 required) to generate a correlation heatmap.")
        return
    
    corr = numeric_df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Numeric Columns', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Add description and conclusion
    st.subheader("What this graph shows:")
    st.markdown("""
    - This heatmap visualizes the correlation matrix for all numeric columns in the dataset.
    - **Correlation** measures the linear relationship between two variables, ranging from -1 to +1.
    - **Warm colors (e.g., red)** indicate a **strong positive correlation** (as one variable increases, the other tends to increase).
    - **Cool colors (e.g., blue)** indicate a **strong negative correlation** (as one variable increases, the other tends to decrease).
    - **Colors near zero (neutral)** indicate a **weak or no linear correlation**.
    - The diagonal is always 1.0 because a variable is perfectly correlated with itself.
    - This is crucial for identifying multicollinearity and understanding relationships between variables.
    """)
    save_and_show_plot(key_suffix="heatmap")

def plot_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str):
    """Generates an interactive scatter plot for two numeric columns."""
    print_header(f"Generating Scatter Plot: '{x_col}' vs '{y_col}'")
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot: {y_col} vs. {x_col}',
                     trendline="ols", trendline_color_override="red")
    
    # Add description and conclusion
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This interactive scatter plot shows the relationship between **`{x_col}`** (on the x-axis) and **`{y_col}`** (on the y-axis).
    - Each point represents a single data entry (row) from your dataset.
    - You can look for patterns such as a positive trend (points going up and to the right), a negative trend (points going down and to the right), or no clear pattern.
    - The **red line** is a **trendline** (calculated using Ordinary Least Squares regression), which shows the general direction of the relationship.
    - Hover over points to see their exact values.
    """)
    st.plotly_chart(fig, use_container_width=True)

def plot_pair_plot(df: pd.DataFrame):
    """Generates a pair plot for numeric columns."""
    print_header("Generating Pair Plot")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns (at least 2 required) for a pair plot.")
        return
    if numeric_df.shape[1] > 5:
        st.warning("Pair plot is limited to 5 numeric columns for performance. Using the first 5.")
        numeric_df = numeric_df.iloc[:, :5]
        
    st.info("Generating pair plot... this may take a moment.")
    fig = sns.pairplot(numeric_df, diag_kind='kde')
    
    # Add description and conclusion
    st.subheader("What this graph shows:")
    st.markdown("""
    - A pair plot provides a matrix of visualizations to show relationships between all pairs of numeric variables.
    - The plots on the **diagonal** are **histograms or KDE plots**, showing the distribution of a single variable.
    - The **off-diagonal plots** are **scatter plots**, showing the relationship between two different variables.
    - This is a powerful tool for getting a quick overview of the relationships and distributions within your numeric data.
    """)
    st.pyplot(fig)

def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str):
    """Generates an interactive time series plot."""
    print_header(f"Generating Time Series Plot: '{value_col}' over '{time_col}'")
    fig = px.line(df, x=time_col, y=value_col, title=f'{value_col} over Time')
    
    # Add description and conclusion
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This line chart plots the values of **`{value_col}`** over time, using the **`{time_col}`** column as the time axis.
    - It is used to identify trends, seasonality, and patterns in your data over a period.
    - You can interact with the plot by zooming and panning to inspect specific time ranges.
    """)
    st.plotly_chart(fig, use_container_width=True)

def handle_missing_values(df: pd.DataFrame, column: str, strategy: str, fill_value=None) -> pd.DataFrame:
    """Handles missing values in a specific column based on the selected strategy."""
    df_processed = df.copy()
    if strategy == "Impute with Mean":
        fill_val = df_processed[column].mean()
        df_processed[column].fillna(fill_val, inplace=True)
        st.success(f"Imputed missing values in '{column}' with mean ({fill_val:.2f}).")
    elif strategy == "Impute with Median":
        fill_val = df_processed[column].median()
        df_processed[column].fillna(fill_val, inplace=True)
        st.success(f"Imputed missing values in '{column}' with median ({fill_val:.2f}).")
    elif strategy == "Impute with Mode":
        fill_val = df_processed[column].mode()[0]
        df_processed[column].fillna(fill_val, inplace=True)
        st.success(f"Imputed missing values in '{column}' with mode ('{fill_val}').")
    elif strategy == "Fill with a specific value":
        if fill_value is not None:
            try:
                # Try to convert fill_value to the column's dtype
                converted_value = pd.Series([fill_value]).astype(df_processed[column].dtype).iloc[0]
                df_processed[column].fillna(converted_value, inplace=True)
                st.success(f"Filled missing values in '{column}' with '{converted_value}'.")
            except (ValueError, TypeError):
                st.error(f"Could not convert '{fill_value}' to the data type of column '{column}'. Please provide a compatible value.")
        else:
            st.warning("Please provide a specific value to fill.")
    elif strategy == "Drop rows with missing values":
        df_processed.dropna(subset=[column], inplace=True)
        st.success(f"Dropped rows with missing values in column '{column}'.")
    
    return df_processed

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate rows from the dataframe."""
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        df_cleaned = df.drop_duplicates()
        st.success(f"Removed {num_duplicates} duplicate rows.")
        return df_cleaned
    else:
        st.info("No duplicate rows found.")
        return df

def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Drops selected columns from the dataframe."""
    df_processed = df.drop(columns=columns_to_drop)
    st.success(f"Dropped columns: {', '.join(columns_to_drop)}.")
    return df_processed

def scale_features(df: pd.DataFrame, columns_to_scale: list, scaler_type: str) -> pd.DataFrame:
    """Scales selected numeric features using StandardScaler or MinMaxScaler."""
    df_processed = df.copy()
    
    if not columns_to_scale:
        st.warning("Please select at least one column to scale.")
        return df_processed

    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
        df_processed[columns_to_scale] = scaler.fit_transform(df_processed[columns_to_scale])
        st.success(f"Applied StandardScaler to columns: {', '.join(columns_to_scale)}.")
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        df_processed[columns_to_scale] = scaler.fit_transform(df_processed[columns_to_scale])
        st.success(f"Applied MinMaxScaler to columns: {', '.join(columns_to_scale)}.")
    
    return df_processed

def plot_3d_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str | None = None):
    """Generates an interactive 3D scatter plot."""
    print_header(f"Generating 3D Scatter Plot: '{x_col}' vs '{y_col}' vs '{z_col}'")
    
    title = f'3D Scatter Plot: {x_col}, {y_col}, {z_col}'
    if color_col and color_col != "None":
        title += f' (Colored by {color_col})'
    else:
        color_col = None # Ensure 'None' string is not passed to plotly

    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, title=title)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # Add description and conclusion
    st.subheader("What this graph shows:")
    description = f"""
    - This interactive 3D scatter plot visualizes the relationship between three numeric variables: **`{x_col}`** (X-axis), **`{y_col}`** (Y-axis), and **`{z_col}`** (Z-axis).
    - Each point represents a single data entry. You can rotate, pan, and zoom the plot to explore the data from different angles.
    """
    if color_col:
        description += f"- The points are colored based on the **`{color_col}`** column, which can help identify clusters or patterns related to that variable."
    
    st.markdown(description)
    st.plotly_chart(fig, use_container_width=True)

def plot_outlier_analysis(df: pd.DataFrame, column: str, outliers: list):
    """Creates a pair plot that highlights outlier data points to analyze their relationships with other variables."""
    if not outliers:
        return

    with st.expander("Detailed Outlier Analysis (Highlighted Pair Plot)"):
        
        # Create a temporary dataframe for plotting
        plot_df = df.copy()
        # Add a new column to identify outliers
        plot_df['is_outlier'] = plot_df[column].isin(outliers).map({True: 'Outlier', False: 'Normal'})
        
        st.markdown("---")
        st.markdown("### ✅ What the Pair Plot Is Showing")
        st.markdown(f"""
        You have a **pair plot** (scatterplot matrix) where:
        - **<span style='color:red;'>Red points</span> = Outliers** (specifically: the **{len(outliers)} outliers** detected in the **`{column}`** column.)
        - **<span style='color:blue;'>Blue points</span> = Normal data points**

        Each cell in the grid shows:
        - A scatter plot of two variables (relationship between them)
        - Or a KDE/density plot (distribution of one variable)
        """, unsafe_allow_html=True)

        # Determine which numeric columns will be plotted for the description and analysis
        numeric_cols_for_analysis = plot_df.select_dtypes(include=np.number).columns.tolist()
        # Exclude the outlier-defining column itself from the interpretation of *other* variables
        if column in numeric_cols_for_analysis:
            numeric_cols_for_analysis.remove(column)

        numeric_cols_for_plot = numeric_cols_for_analysis[:] # Copy for plotting, might be truncated

        if len(numeric_cols_for_plot) > 5:
            st.warning("Pair plot is limited to 5 numeric columns for performance. Using the first 5 for visualization.")
            numeric_cols_for_plot = numeric_cols_for_plot[:5]
        
        st.markdown(f"The variables plotted include: **{', '.join(numeric_cols_for_plot)}**")
        st.markdown("---")

        # Generate the plot
        # Ensure 'is_outlier' is included for hue
        cols_for_pairplot = [col for col in numeric_cols_for_plot if col in plot_df.columns] + ['is_outlier']
        
        if len(numeric_cols_for_plot) < 1: # Need at least one other numeric column to compare
            st.warning("Not enough other numeric columns to create a pair plot for outlier analysis.")
            return

        with st.spinner("Generating outlier pair plot..."):
            fig = sns.pairplot(plot_df[cols_for_pairplot], hue='is_outlier', palette={'Normal': 'blue', 'Outlier': 'red'}, corner=True)
            st.pyplot(fig)

        # Dynamic Interpretation Logic
        outlier_df_subset = plot_df[plot_df['is_outlier'] == 'Outlier']
        normal_df_subset = plot_df[plot_df['is_outlier'] == 'Normal']
        
        behavioral_outlier_found = False
        interpretation_details = []

        if not outlier_df_subset.empty and not normal_df_subset.empty:
            for col_to_compare in numeric_cols_for_analysis: # Use full list for analysis
                if col_to_compare in outlier_df_subset.columns and col_to_compare in normal_df_subset.columns:
                    outlier_mean = outlier_df_subset[col_to_compare].mean()
                    normal_mean = normal_df_subset[col_to_compare].mean()
                    normal_std = normal_df_subset[col_to_compare].std()

                    # Heuristic: if outlier mean is more than 1.5 std dev away from normal mean
                    if normal_std > 0 and abs(outlier_mean - normal_mean) > 1.5 * normal_std:
                        behavioral_outlier_found = True
                        if outlier_mean > normal_mean:
                            interpretation_details.append(f"- For **`{col_to_compare}`**, outliers tend to have **higher values** (mean: {outlier_mean:.2f} vs normal mean: {normal_mean:.2f}).")
                        else:
                            interpretation_details.append(f"- For **`{col_to_compare}`**, outliers tend to have **lower values** (mean: {outlier_mean:.2f} vs normal mean: {normal_mean:.2f}).")
            
            st.markdown("### How to interpret YOUR specific plot")
            if behavioral_outlier_found:
                st.markdown("""
                Based on the analysis of the generated plot and statistical comparison:
                - ✔ The red points (outliers) show **distinct patterns or tendencies** when compared to the blue points (normal data) across some variables.
                - ✔ This suggests that these outliers might represent a **meaningful subgroup** or a special type of user/event.
                """)
                st.markdown("### Interpretation")
                st.markdown(f"""
                - The outliers in **`{column}`** appear to be **behavioral outliers**, indicating they are part of a potentially distinct group.
                - Their unusual behavior is reflected in:
                """)
                for detail in interpretation_details:
                    st.markdown(detail)
                st.markdown("""
                - Further investigation into these specific characteristics could reveal valuable insights.
                """)
            else:
                st.markdown(f"""
                Based on the analysis of the generated plot:
                - ✔ Almost all red points are mixed uniformly with the blue points.
                - ✔ There is **no clear cluster**, no separate region, no special trend.
                - ✔ In every scatter cell, red + blue overlap heavily.
                - ✔ KDE curves (the top diagonal) show no distinct second peak for outliers.

                This means:
                ### Interpretation
                - The outliers in **`{column}`** are *not a separate user group* based on their relationship with other numeric variables.
                - They don't behave significantly differently in the other plotted variables.
                - They seem to be **statistical outliers only**, not behavioral outliers.

                In simple words: **The outliers don’t form a special pattern. They’re just extreme values, not a different type of user.**
                """, unsafe_allow_html=True)
        else:
            st.markdown("### How to interpret YOUR specific plot")
            st.markdown("Not enough data points (either outliers or normal data) to perform a comparative interpretation.")


        st.markdown("---")

        st.markdown("### What this visualization helps you understand")
        st.markdown(f"""
        The purpose of this pair plot is to answer **one key question**:
        ### *“Do the outliers behave differently from normal data?”*

        If the red points (outliers) form **clusters**, **patterns**, or **separate shapes**, it means:
        - There's likely a meaningful subgroup or hidden segment
        - The outliers may not be “errors” but represent a special type of user
        - Their unusual behavior might be systematic and worth analyzing

        If the red points **do NOT form clusters**, and instead are spread randomly:
        - The outliers are probably statistical noise
        - They don’t represent a distinct group
        - They may just be extreme values of normal users
        """)
        st.markdown("---")

        st.markdown("### Why does this matter?")
        if behavioral_outlier_found:
            st.markdown("""
            Because it affects how you treat those outliers:
            - ✔ **If they formed a cluster (your case) → keep them; they represent a meaningful subgroup.**
            - ❌ If they don’t → you can safely remove or cap them

            This improves:
            - Model accuracy
            - Distribution normality
            - Statistical interpretations
            """)
        else:
            st.markdown("""
            Because it affects how you treat those outliers:
            - ✔ If they formed a cluster → keep them; they represent a meaningful subgroup
            - ❌ **If they don’t (your case) → you can safely remove or cap them.**

            This improves:
            - Model accuracy
            - Distribution normality
            - Statistical interpretations
            """)
        st.markdown("---")