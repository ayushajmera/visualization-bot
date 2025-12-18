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

# --- Visualization theme / palette management ---
# Default palette mapping for both seaborn (matplotlib) and Plotly
PALETTES = {
    'Colorblind': {'sns': 'colorblind', 'plotly': px.colors.qualitative.Safe},
    'Pastel': {'sns': 'pastel', 'plotly': px.colors.qualitative.Pastel},
    'Muted': {'sns': 'muted', 'plotly': px.colors.qualitative.Plotly},
    'Deep': {'sns': 'deep', 'plotly': px.colors.qualitative.Dark24 if hasattr(px.colors.qualitative, 'Dark24') else px.colors.qualitative.Plotly},
    'Viridis': {'sns': 'viridis', 'plotly': px.colors.sequential.Viridis}
}

# Current palette state
_CURRENT_PALETTE_NAME = 'Colorblind'
_CURRENT_PLOTLY_SEQ = PALETTES[_CURRENT_PALETTE_NAME]['plotly']

# Initialize seaborn theme with default palette
sns.set_theme(style='whitegrid', palette=PALETTES[_CURRENT_PALETTE_NAME]['sns'])


def set_color_palette(palette_name: str):
    """Set the color palette for seaborn/matplotlib and Plotly.

    palette_name must be one of the keys in PALETTES.
    When changed, this will **clear Streamlit caches** so that cached visualizations are regenerated with the new palette.
    """
    global _CURRENT_PALETTE_NAME, _CURRENT_PLOTLY_SEQ
    if palette_name not in PALETTES:
        st.warning(f"Palette '{palette_name}' not recognized. Using default '{_CURRENT_PALETTE_NAME}'.")
        return
    _CURRENT_PALETTE_NAME = palette_name
    mapping = PALETTES[palette_name]
    # Update seaborn theme
    try:
        sns.set_theme(style='whitegrid', palette=mapping['sns'])
    except Exception:
        # Fallback: use default seaborn theme
        sns.set_theme(style='whitegrid')
    # Update plotly color sequence
    _CURRENT_PLOTLY_SEQ = mapping['plotly']

    # Clear Streamlit caches so visuals regenerate using the new palette
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass


def get_plotly_color_sequence():
    return _CURRENT_PLOTLY_SEQ

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
            df = pd.read_excel(uploaded_file, engine=None) # Let pandas auto-detect
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Error: Unsupported file format '{file_extension}'. Please use CSV, Excel, or JSON.")
            return None

        st.success("Dataset loaded successfully!")
        st.write("--- Dataset Shape ---")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.write("--- Columns and Data Types ---")
        st.dataframe(df.dtypes.astype(str).to_frame('Data Type'))
        
        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

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

@st.cache_resource(show_spinner="Generating histogram...")
def plot_histogram(df: pd.DataFrame, column: str, outliers_to_plot: list | None = None, outliers_summary_count: int | None = None):
    """Generates and saves a histogram, highlighting outliers if provided.

    Parameters:
        df: DataFrame containing the data.
        column: Column name to plot.
        outliers_to_plot: Optional list of numeric outlier values to mark on the plot.
        outliers_summary_count: Optional integer to display an outlier summary on the plot.
    """
    # Explicitly create a figure and axes object
    fig, ax = plt.subplots(figsize=(12, 7))
    # Use the first color from the current seaborn palette for the histogram
    palette_colors = sns.color_palette()
    hist_color = palette_colors[0] if palette_colors else None
    sns.histplot(df[column], kde=True, bins=30, ax=ax, color=hist_color)
    
    # If outliers are provided, mark them on the histogram
    if outliers_to_plot:
        try:
            for val in outliers_to_plot:
                # Draw a dashed red vertical line for each outlier
                ax.axvline(val, color='red', linestyle='--', linewidth=1)
            # Add a small legend/annotation showing the number of outliers, if provided
            if outliers_summary_count is not None:
                ax.text(0.95, 0.95, f'Outliers: {outliers_summary_count}', transform=ax.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        except Exception as e:
            # Fail gracefully and still return the histogram
            st.warning(f"Could not annotate outliers on histogram: {e}")
    ax.set_title(f'Histogram of {column}', fontsize=16)
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
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig, buf

@st.cache_resource(show_spinner="Generating bar chart...")
def plot_bar_chart(df: pd.DataFrame, column: str, max_bars: int = 20, **kwargs): 
    """Generates and saves a bar chart for a categorical column.

    To keep charts readable, only the top `max_bars` categories are shown; the rest are grouped into an 'Other' bucket.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use value_counts and handle grouping if too many categories
    counts = df[column].value_counts()

    if counts.empty:
        st.warning(f"Column '{column}' contains no values to plot.")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return fig, buf

    if len(counts) > max_bars:
        st.warning(f"Column '{column}' has {len(counts)} categories — showing top {max_bars} and grouping the rest as 'Other'.")
        top_counts = counts.nlargest(max_bars)
        other_sum = counts.drop(top_counts.index).sum()
        counts = top_counts
        if other_sum > 0:
            counts['Other'] = other_sum

    counts = counts.sort_values(ascending=False)
    colors = sns.color_palette(n_colors=len(counts))

    # Dynamic sizing and explicit bar width for predictable rendering across counts
    num_bars = len(counts)
    if num_bars <= 10:
        fig.set_size_inches(max(8, num_bars * 0.9), 6)
        bar_width = 0.6
    elif num_bars <= 20:
        fig.set_size_inches(max(10, num_bars * 0.6), 6.5)
        bar_width = 0.6
    else:
        fig.set_size_inches(max(12, num_bars * 0.45), 7)
        bar_width = max(0.35, min(0.8, 10.0 / num_bars))

    # Draw bars with Matplotlib directly for predictable rendering and visible edges
    x_positions = np.arange(num_bars)
    ax.bar(x_positions, counts.values, width=bar_width, color=colors, edgecolor='black', linewidth=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(counts.index, rotation=45, ha='right')

    # Ensure axes limits include the bars clearly
    ax.set_xlim(-0.5, num_bars - 0.5 + (1 - bar_width))
    ax.set_ylim(0, counts.values.max() * 1.15 if counts.values.max() > 0 else 1)

    ax.set_title(f'Bar Chart of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    ax.margins(x=0.01)
    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig, buf

@st.cache_resource(show_spinner="Generating pie chart...")
def plot_pie_chart(df: pd.DataFrame, column: str, max_slices: int = 10):
    """Generates and saves a pie chart for a categorical column.

    To avoid creating many microscopically thin slices, we limit the chart to the top
    `max_slices` categories and group all remaining categories into an 'Other' slice.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    counts = df[column].value_counts()

    if counts.empty:
        st.warning(f"Column '{column}' contains no values to plot.")
        # Return an empty figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return fig, buf

    # If there are more categories than max_slices, keep only the top categories and group the rest
    if len(counts) > max_slices:
        st.warning(f"Column '{column}' has {len(counts)} unique categories — limiting pie chart to the top {max_slices} and grouping the rest as 'Other'.")
        top_counts = counts.nlargest(max_slices)
        other_sum = counts.drop(top_counts.index).sum()
        counts = top_counts
        if other_sum > 0:
            counts['Other'] = other_sum

    # Sort counts so the largest slices come first for better readability
    counts = counts.sort_values(ascending=False)

    # Draw the pie chart with percentages and a tidy layout
    colors = sns.color_palette(n_colors=len(counts))
    ax.pie(counts, labels=counts.index, autopct=lambda pct: f"{pct:.1f}%" if pct >= 1 else '', startangle=140, wedgeprops={'edgecolor': 'white'}, colors=colors)
    ax.set_title(f'Pie Chart of {column}', fontsize=16)
    plt.ylabel('') # Hide the y-label which is often the column name
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig, buf

@st.cache_resource(show_spinner="Generating box plot...")
def plot_box_plot(df: pd.DataFrame, column: str):
    """Generates and saves a box plot for a numeric column."""
    fig, ax = plt.subplots(figsize=(12, 7))
    palette_colors = sns.color_palette()
    box_color = palette_colors[0] if palette_colors else None
    sns.boxplot(x=df[column], ax=ax, color=box_color)
    ax.set_title(f'Box Plot of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.grid(axis='x', alpha=0.5)
    
    # Add description and conclusion
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig, buf

@st.cache_resource(show_spinner="Generating correlation heatmap...")
def plot_correlation_heatmap(df: pd.DataFrame):
    """Calculates and plots the correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns (at least 2 required) to generate a correlation heatmap.")
        return None, None # Return None for both figure and buffer
    
    corr = numeric_df.corr()
    fig = plt.figure(figsize=(14, 10)) # Explicitly create the figure
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Numeric Columns', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the explicitly created figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig, buf

@st.cache_data(show_spinner="Generating scatter plot...")
def plot_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str):
    """Generates an interactive scatter plot for two numeric columns."""
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot: {y_col} vs. {x_col}',
                     trendline="ols", trendline_color_override="red",
                     color_discrete_sequence=get_plotly_color_sequence(), template='plotly_white')
    
    # Add description and conclusion
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This interactive scatter plot shows the relationship between **`{x_col}`** (on the x-axis) and **`{y_col}`** (on the y-axis).
    - Each point represents a single data entry (row) from your dataset.
    - You can look for patterns such as a positive trend (points going up and to the right), a negative trend (points going down and to the right), or no clear pattern.
    - The **red line** is a **trendline** (calculated using Ordinary Least Squares regression), which shows the general direction of the relationship.
    - Hover over points to see their exact values.
    """)
    # Note: Do not render the chart here; return the Plotly Figure for the caller to display.
    return fig

@st.cache_resource(show_spinner="Generating pair plot...")
def plot_pair_plot(df: pd.DataFrame):
    """Generates a pair plot for numeric columns."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns (at least 2 required) for a pair plot.")
        return None
    if numeric_df.shape[1] > 5:
        st.warning("Pair plot is limited to 5 numeric columns for performance. Using the first 5.")
        numeric_df = numeric_df.iloc[:, :5]
        
    st.info("Generating pair plot... this may take a moment.")
    # Respect the current seaborn palette
    try:
        pair_palette = sns.color_palette()
        fig = sns.pairplot(numeric_df, diag_kind='kde', palette=pair_palette)
    except Exception:
        fig = sns.pairplot(numeric_df, diag_kind='kde')
    
    # Pairplot returns a Figure object, not a buffer directly. We handle it in the UI.
    return fig

@st.cache_resource(show_spinner="Generating categorical box plot...")
def plot_categorical_boxplot(df: pd.DataFrame, cat_col: str, num_col: str):
    """Generates a box plot of a numeric column grouped by a categorical column."""
    # This function does not need a print_header as it's part of a larger section.
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For readability, limit to the top 15 most frequent categories
    top_categories = df[cat_col].value_counts().nlargest(15).index
    df_filtered = df[df[cat_col].isin(top_categories)]
    
    sns.boxplot(data=df_filtered, x=cat_col, y=num_col, ax=ax)
    
    ax.set_title(f'Distribution of {num_col} across {cat_col} Categories', fontsize=16)
    plt.xlabel(cat_col, fontsize=12)
    plt.ylabel(num_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return fig, buf

@st.cache_data(show_spinner="Generating time series plot...")
def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str):
    """Generates an interactive time series plot."""
    print_header(f"Generating Time Series Plot: '{value_col}' over '{time_col}'")
    fig = px.line(df, x=time_col, y=value_col, title=f'{value_col} over Time', color_discrete_sequence=get_plotly_color_sequence(), template='plotly_white')
    
    # Add description and conclusion
    st.subheader("What this graph shows:")
    st.markdown(f"""
    - This line chart plots the values of **`{value_col}`** over time, using the **`{time_col}`** column as the time axis.
    - It is used to identify trends, seasonality, and patterns in your data over a period.
    - You can interact with the plot by zooming and panning to inspect specific time ranges.
    """)
    st.plotly_chart(fig, width='stretch')

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

@st.cache_data(show_spinner="Generating 3D scatter plot...")
def plot_3d_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str | None = None):
    """Generates an interactive 3D scatter plot."""
    print_header(f"Generating 3D Scatter Plot: '{x_col}' vs '{y_col}' vs '{z_col}'")
    
    title = f'3D Scatter Plot: {x_col}, {y_col}, {z_col}'
    if color_col and color_col != "None":
        title += f' (Colored by {color_col})'
    else:
        color_col = None # Ensure 'None' string is not passed to plotly

    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col if color_col else None, title=title,
                         color_discrete_sequence=get_plotly_color_sequence(), template='plotly_white')
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
    st.plotly_chart(fig, width='stretch')

@st.cache_resource(show_spinner="Generating outlier pair plot...")
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
        st.markdown("### What the Pair Plot Is Showing")
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
                - The red points (outliers) show **distinct patterns or tendencies** when compared to the blue points (normal data) across some variables.
                - This suggests that these outliers might represent a **meaningful subgroup** or a special type of user/event.
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
                - Almost all red points are mixed uniformly with the blue points.
                - There is **no clear cluster**, no separate region, no special trend.
                - In every scatter cell, red + blue overlap heavily.
                - KDE curves (the top diagonal) show no distinct second peak for outliers.

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
            - If they formed a cluster (your case) → keep them; they represent a meaningful subgroup.
            - If they don’t → you can safely remove or cap them

            This improves:
            - Model accuracy
            - Distribution normality
            - Statistical interpretations
            """)
        else:
            st.markdown("""
            Because it affects how you treat those outliers:
            - If they formed a cluster → keep them; they represent a meaningful subgroup
            - If they don’t (your case) → you can safely remove or cap them.

            This improves:
            - Model accuracy
            - Distribution normality
            - Statistical interpretations
            """)
        st.markdown("---")

def correct_data_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """Converts a column to a specified data type, handling errors."""
    df_processed = df.copy()
    
    if column not in df_processed.columns:
        st.error(f"Column '{column}' not found in the DataFrame.")
        return df # Return original df

    st.write(f"Attempting to convert column **`{column}`** to **`{target_type}`**...")

    try:
        original_nulls = df_processed[column].isnull().sum()
        if target_type == 'numeric':
            # This will convert to int or float as appropriate, coercing errors to NaN
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
        elif target_type == 'datetime':
            df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
        elif target_type == 'category':
            df_processed[column] = df_processed[column].astype('category')
        elif target_type == 'string':
            df_processed[column] = df_processed[column].astype(str)
        else:
            st.error(f"Unsupported target type '{target_type}'.")
            return df # Return original df

        st.success(f"Successfully converted '{column}' to {target_type}.")
        new_nulls = df_processed[column].isnull().sum()
        if new_nulls > original_nulls:
            st.warning(f"{new_nulls - original_nulls} values could not be converted and were set to null (NaN/NaT).")

    except Exception as e:
        st.error(f"An error occurred during conversion of column '{column}': {e}")
        return df # Return original df on failure

    return df_processed

@st.cache_data(show_spinner="Standardizing categories...")
def standardize_categories(df: pd.DataFrame, column: str, values_to_merge: list, new_value: str) -> pd.DataFrame:
    """Standardizes categories by merging multiple values into a single new value."""
    df_processed = df.copy()
    
    if column not in df_processed.columns:
        st.error(f"Column '{column}' not found.")
        return df

    try:
        df_processed[column] = df_processed[column].replace(values_to_merge, new_value)
        st.success(f"Standardized {len(values_to_merge)} categories into '{new_value}' in column '{column}'.")
    except Exception as e:
        st.error(f"An error occurred during standardization: {e}")
        return df

    return df_processed

@st.cache_data(show_spinner="Creating derived numeric column...")
def create_derived_column_numeric(df: pd.DataFrame, col_a: str, operation: str, col_b: str, new_col_name: str) -> pd.DataFrame:
    """Creates a new column by performing an operation on two numeric columns."""
    df_processed = df.copy()

    if new_col_name in df_processed.columns:
        st.error(f"Column '{new_col_name}' already exists. Please choose a different name.")
        return df

    try:
        if operation == '+':
            df_processed[new_col_name] = df_processed[col_a] + df_processed[col_b]
        elif operation == '-':
            df_processed[new_col_name] = df_processed[col_a] - df_processed[col_b]
        elif operation == '*':
            df_processed[new_col_name] = df_processed[col_a] * df_processed[col_b]
        elif operation == '/':
            # Add a small epsilon to avoid division by zero
            df_processed[new_col_name] = df_processed[col_a] / (df_processed[col_b] + 1e-9)
        
        st.success(f"Created new column '{new_col_name}' as `{col_a} {operation} {col_b}`.")
    except Exception as e:
        st.error(f"An error occurred while creating the derived column: {e}")
        return df

    return df_processed

@st.cache_data(show_spinner="Extracting datetime features...")
def create_derived_column_datetime(df: pd.DataFrame, time_col: str, parts_to_extract: list) -> pd.DataFrame:
    """Creates new columns by extracting parts from a datetime column."""
    df_processed = df.copy()

    try:
        # Ensure the column is in datetime format, coercing errors
        datetime_series = pd.to_datetime(df_processed[time_col], errors='coerce')

        for part in parts_to_extract:
            new_col_name = f"{time_col}_{part.lower().replace(' ', '_')}"
            if part == "Year":
                df_processed[new_col_name] = datetime_series.dt.year
            elif part == "Month":
                df_processed[new_col_name] = datetime_series.dt.month
            elif part == "Day":
                df_processed[new_col_name] = datetime_series.dt.day
            elif part == "Quarter":
                df_processed[new_col_name] = datetime_series.dt.quarter
            elif part == "Day of Week":
                df_processed[new_col_name] = datetime_series.dt.dayofweek
            elif part == "Day Name":
                df_processed[new_col_name] = datetime_series.dt.day_name()
            elif part == "Week of Year":
                df_processed[new_col_name] = datetime_series.dt.isocalendar().week
        
        st.success(f"Extracted {', '.join(parts_to_extract)} from '{time_col}'.")
    except Exception as e:
        st.error(f"An error occurred while extracting datetime parts: {e}")
        return df

    return df_processed

@st.cache_data(show_spinner="Creating binned column...")
def create_binned_column(df: pd.DataFrame, col_to_bin: str, num_bins: int, new_col_name: str) -> pd.DataFrame:
    """Creates a new categorical column by binning a numeric column."""
    df_processed = df.copy()
    
    if new_col_name in df_processed.columns:
        st.error(f"Column '{new_col_name}' already exists. Please choose a different name.")
        return df

    try:
        df_processed[new_col_name] = pd.cut(df_processed[col_to_bin], bins=num_bins, labels=False, include_lowest=True)
        st.success(f"Created binned column '{new_col_name}' from '{col_to_bin}' with {num_bins} bins.")
    except Exception as e:
        st.error(f"An error occurred during binning: {e}")
        return df

    return df_processed

@st.cache_data(show_spinner="Generating grouped analysis...")
def plot_grouped_analysis(df: pd.DataFrame):
    """
    Performs grouped analysis by plotting the mean of numeric columns against top categories of a categorical column.
    """
    print_header("Grouped Analysis (Mean of Numeric by Category)")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not categorical_cols:
        st.info("No categorical columns found for grouped analysis.")
        return
    if not numeric_cols:
        st.info("No numeric columns found for grouped analysis.")
        return

    # Heuristic: Pick a categorical column with a reasonable number of unique values (2-20 is ideal)
    cat_col_to_use = None
    for col in categorical_cols:
        if 2 <= df[col].nunique() <= 20:
            cat_col_to_use = col
            break
    
    if not cat_col_to_use: # Fallback if no ideal column is found
        cat_col_to_use = categorical_cols[0]

    st.markdown(f"Showing the average of numeric columns, grouped by the **`{cat_col_to_use}`** category. This helps to see how numeric values differ across different groups.")
    
    # Limit to top 20 categories for readability
    top_categories = df[cat_col_to_use].value_counts().nlargest(20).index
    df_filtered = df[df[cat_col_to_use].isin(top_categories)]

    for num_col in numeric_cols:
        with st.expander(f"Mean of `{num_col}` by `{cat_col_to_use}`"):
            try:
                fig = px.bar(df_filtered.groupby(cat_col_to_use)[num_col].mean().reset_index(), 
                             x=cat_col_to_use, y=num_col, title=f'Mean of {num_col} by {cat_col_to_use}',
                             color_discrete_sequence=get_plotly_color_sequence(), template='plotly_white')
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Could not generate grouped plot for '{num_col}': {e}")

@st.cache_data(show_spinner="Generating faceted scatter plot...")
def plot_faceted_scatter(df: pd.DataFrame):
    """
    Generates a faceted scatter plot to show the relationship between two numeric variables across different categories.
    """
    print_header("Faceted Scatter Analysis (2 Numerics by 1 Category)")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns (at least 2 required) for a faceted scatter plot.")
        return
    if not categorical_cols:
        st.info("No categorical columns found for a faceted scatter plot.")
        return

    # --- Heuristics for column selection ---
    # 1. Find a suitable categorical column (low cardinality)
    cat_col_to_use = None
    for col in categorical_cols:
        if 2 <= df[col].nunique() <= 10:
            cat_col_to_use = col
            break
    if not cat_col_to_use: # Fallback
        cat_col_to_use = categorical_cols[0]

    # 2. Find the two most correlated numeric columns
    corr_matrix = df[numeric_cols].corr().abs()
    sol = corr_matrix.unstack()
    so = sol.sort_values(kind="quicksort", ascending=False)
    so = so[so < 1] # Remove self-correlations
    if so.empty:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
    else:
        x_col, y_col = so.index[0]

    st.markdown(f"This plot shows the relationship between the two most correlated numeric variables, **`{x_col}`** and **`{y_col}`**, broken down by the categories in **`{cat_col_to_use}`**.")
    st.markdown("This helps to see if the relationship between the two numeric variables is consistent across different groups.")

    try:
        # Handle high-cardinality categorical columns: cap number of facets to top N categories
        max_facets = 20
        df_for_plot = df.copy()
        actual_nuniques = df_for_plot[cat_col_to_use].nunique()
        if actual_nuniques > max_facets:
            st.warning(f"Column '{cat_col_to_use}' has {actual_nuniques} categories — limiting to top {max_facets} and grouping the rest as 'Other' for readability.")
            top_cats = df_for_plot[cat_col_to_use].value_counts().nlargest(max_facets).index
            df_for_plot[cat_col_to_use] = df_for_plot[cat_col_to_use].where(df_for_plot[cat_col_to_use].isin(top_cats), other='Other')

        # Choose a sensible facet wrap and a small row spacing to avoid Plotly spacing errors
        facet_wrap = 4
        facet_row_spacing = 0.01

        fig = px.scatter(df_for_plot, x=x_col, y=y_col, facet_col=cat_col_to_use, facet_col_wrap=facet_wrap,
                         facet_row_spacing=facet_row_spacing,
                         title=f'Scatter plot of {x_col} vs {y_col}, Faceted by {cat_col_to_use}',
                         color_discrete_sequence=get_plotly_color_sequence(), template='plotly_white')
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.error(f"Could not generate faceted scatter plot: {e}")

@st.cache_data(show_spinner="Calculating segmentation")
def _calculate_pareto_analysis(df: pd.DataFrame, id_col: str, value_col: str):
    """
    Internal cached function to perform the heavy lifting of Pareto analysis.
    """
    grouped_df = df.groupby(id_col)[value_col].sum().sort_values(ascending=False).reset_index()
    grouped_df['cumulative_value'] = grouped_df[value_col].cumsum()
    total_value = grouped_df[value_col].sum()
    if total_value == 0: # Avoid division by zero
        grouped_df['cumulative_percentage'] = 0
    else:
        grouped_df['cumulative_percentage'] = (grouped_df['cumulative_value'] / total_value) * 100

    # Create segments
    pareto_threshold = 80
    top_segment = grouped_df[grouped_df['cumulative_percentage'] <= pareto_threshold]
    bottom_segment = grouped_df[grouped_df['cumulative_percentage'] > pareto_threshold]
    return top_segment, bottom_segment, len(grouped_df)

def perform_segmentation_analysis(df: pd.DataFrame):
    """
    Performs segmentation analysis, focusing on a Pareto (80/20) analysis of customers or entities.
    Handles UI for column selection and displays results.
    """
    print_header("Segmentation & Grouping (Pareto Analysis)")

    run_analysis = False

    # --- Heuristics to find ID and Value columns ---
    id_col = None
    value_col = None

    # Find a potential ID column (high unique values)
    for col in df.columns:
        if 'id' in col.lower() or 'customer' in col.lower() or 'user' in col.lower():
            if df[col].nunique() / len(df) > 0.5: # Likely an identifier
                id_col = col
                break

    # Find a potential value column (numeric, positive values)
    for col in df.select_dtypes(include=np.number).columns:
        if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'price', 'amount', 'value']):
            if (df[col] >= 0).all(): # Values are generally non-negative
                value_col = col
                break

    # --- Manual Fallback if automatic detection fails ---
    if not id_col or not value_col:
        st.info("Could not automatically identify suitable ID and Value columns (e.g., 'CustomerID' and 'Sales') for segmentation analysis.")
        st.warning("Please manually select the columns for the analysis below.")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Create two columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            id_col = st.selectbox("Select the ID column (e.g., CustomerID, UserID):", options=all_cols)
        with col2:
            value_col = st.selectbox("Select the Value column (e.g., Sales, Revenue):", options=numeric_cols)
        
        if st.button("Run Manual Segmentation Analysis"):
            run_analysis = True
    else:
        run_analysis = True

    if run_analysis and id_col and value_col:
        st.markdown(f"Performing a Pareto analysis by segmenting **`{id_col}`** based on their total **`{value_col}`**.")
        st.markdown("The goal is to identify the vital few (e.g., top 20% of customers) who contribute the most value.")
        st.markdown("---")

        # --- Perform Segmentation by calling the cached function ---
        top_segment, bottom_segment, num_total = _calculate_pareto_analysis(df, id_col, value_col)

        if top_segment is not None and num_total > 0:
            num_top = len(top_segment)
            percent_top_customers = (num_top / num_total) * 100
            pareto_threshold = 80

            st.subheader("Insights Generated")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("Top Segment (The Vital Few)")
                st.metric(label=f"Top {percent_top_customers:.1f}% of '{id_col}'s", value=f"{num_top} entities")
                st.metric(label=f"Contribute to Value", value=f"{pareto_threshold}% of Total {value_col}")

            with col2:
                st.warning("Bottom Segment (The Trivial Many)")
                st.metric(label=f"Bottom {100-percent_top_customers:.1f}% of '{id_col}'s", value=f"{len(bottom_segment)} entities")
                st.metric(label=f"Contribute to Value", value=f"{100-pareto_threshold}% of Total '{value_col}'")

            st.markdown("### Where to focus business efforts:")
            st.markdown(f"- The **Top Segment** represents your most valuable customers/entities. Focus retention, up-selling, and premium services on this group.")
            st.markdown(f"- The **Bottom Segment** is less impactful. Efforts here could focus on automated marketing or identifying potential for growth.")

            # --- Visualization ---
            st.subheader("Segment Value Distribution")
            fig = px.bar(x=['Top Segment', 'Bottom Segment'], 
                         y=[top_segment[value_col].sum(), 0],
                         title=f"Total '{value_col}' by Segment", labels={'x': 'Segment', 'y': f'Total {value_col}'},
                         color_discrete_sequence=get_plotly_color_sequence(), template='plotly_white')
            st.plotly_chart(fig, width='stretch')