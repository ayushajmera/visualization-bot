# Automated Analysis Bot

An interactive web application built with Streamlit that empowers users to perform end-to-end data analysis. Upload a dataset and choose between a fully automated "One-Click" analysis or a guided, step-by-step workflow for preprocessing and custom visualization.

##  Features

### 1. Workflows
- **One-Click Data Analyst**: A "zero-config" option that runs a complete analysis pipeline on your dataset. It performs a data quality audit, generates automated text insights, suggests cleaning steps, and creates a full suite of EDA and time-series visualizations.
- **Guided Analysis**: A step-by-step process for users who want more control.
    - **Data Preprocessing**: A comprehensive toolkit to clean and prepare your data.
    - **Custom Analysis**: A flexible module to generate specific reports and visualizations.

<p align="center">
  <img src="https://i.imgur.com/your-app-demo.gif" alt="App Demo">
</p>

---

### 2. Data Loading
- **Upload Various Formats**: Load datasets from CSV, Excel (`.xlsx`, `.xls`), and JSON files.
- **Instant Feedback**: Get immediate information on dataset shape, column names, and data types upon upload.

### 3. Data Preprocessing
A complete suite of tools to clean and prepare your data for analysis.
- **Drop Unnecessary Columns**: Select and remove irrelevant columns.
- **Handle Missing Values**:
    - Target specific columns with missing data.
    - Choose from multiple imputation strategies (Mean, Median, Mode) or fill with a specific value.
    - Option to drop rows with missing values.
- **Remove Duplicate Rows**: Identify and remove duplicate entries from the dataset.
- **Feature Scaling**:
    - Apply `StandardScaler` or `MinMaxScaler` to selected numeric columns to normalize their range.
- **Correct Data Types**: Manually change the data type of any column (e.g., from `object` to `datetime`).
- **Standardize Categories**: Merge multiple categorical values into a single, standardized value (e.g., mapping `USA` and `United States` to `USA`).
- **Create Derived Columns**:
    - **Numeric Operations**: Create a new column from two existing numeric columns (e.g., `Revenue - Cost` to create `Profit`).
    - **Datetime Extraction**: Extract features like `Year`, `Month`, `Day`, or `Day of Week` from a datetime column.
    - **Binning**: Convert a continuous numeric column into discrete bins or groups (e.g., `Age` into `Age Group`).

### 4. Data Analysis & Visualization
A powerful set of tools to explore and visualize your cleaned data.
- **Automated Insights**:
    - **Data Quality Audit**: Automatically checks for missing values, duplicates, data type issues, and outliers.
    - **Text-Based Summary**: Generates a narrative that answers key questions about the dataset, such as its size, time period, and granularity.
    - **Cleaning Suggestions**: Provides actionable recommendations for data preprocessing.

- **Full Insights Report (on a single column)**:
    - Generate a detailed statistical summary for any numeric column (mean, median, std dev, etc.).
    - Perform multi-method anomaly detection (IQR, Z-Score, Isolation Forest).
    - **Dynamic Outlier Analysis**: A unique pair plot that highlights outliers against normal data points, with a dynamic, data-driven interpretation to determine if outliers are statistical noise or a distinct behavioral group.
- **Univariate Analysis**:
    - **Numeric**: Histograms (with KDE) and Box Plots.
    - **Categorical**: Bar Charts and Pie Charts.
- **Multivariate Analysis**:
    - **Correlation Heatmap**: Visualize the correlation matrix of all numeric columns.
    - **Pair Plot**: Get a quick overview of the relationships and distributions between numeric variables.
    - **Custom 2D Scatter Plot**: Interactive plot for any two numeric variables with a trendline.
    - **Custom 3D Scatter Plot**: Interactive 3D plot to visualize the relationship between three numeric variables, with an optional color dimension.
- **Temporal Analysis**:
    - **Automated Analysis**: Automatically detects time columns, resamples data, and performs seasonal decomposition to identify trends and weekly/monthly patterns.
    - **Custom Plotting**: Manually select time and value columns to plot a time-series line chart.

##  How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd visualization-bot
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

The application will open in your default web browser.

## Deployment to Streamlit Community Cloud

This application can be easily deployed for free using Streamlit Community Cloud.

1.  **Push your project to a public GitHub repository.**
    Make sure your repository includes all project files:
    - `streamlit_app.py` (your main app file)
    - `backend.py` and `full_report.py`
    - `requirements.txt` (listing all dependencies)

2.  **Sign up for Streamlit Community Cloud:**
    - Go to share.streamlit.io and sign up using your GitHub account.

3.  **Deploy the app:**
    - From your workspace, click the "**New app**" button.
    - Select the repository, branch (e.g., `main`), and set the "**Main file path**" to `streamlit_app.py`.
    - Click "**Deploy!**".

Streamlit will then build and deploy your application, making it accessible via a public URL.

## File Structure

The project is organized into a modular structure for clarity and maintainability:

- **`streamlit_app.py`**: The main entry point and UI layer. It controls the sidebar, workflow selection, and calls the appropriate modules.
- **`one_click_analyst.py`**: Orchestrates the "One-Click Data Analyst" workflow by calling various analysis modules in sequence.
- **`Backend.py`**: A reusable "toolbox" containing all the core, low-level functions for data manipulation (e.g., `load_dataset`), plotting (e.g., `plot_histogram`), and calculations (e.g., `scale_features`).
- **Analysis Modules**: Each file focuses on a specific part of the analysis.
    - `data_audit.py`: Performs data quality checks.
    - `automated_insights.py`: Generates text-based summaries.
    - `automated_cleaning.py`: Provides cleaning suggestions.
    - `exploratory_data_analysis.py`: Manages the generation of EDA plots.
    - `time_series_analysis.py`: Handles automated temporal analysis.
    - `full_report.py`: Manages the detailed, single-column insights report.
- **`requirements.txt`**: A list of all Python libraries required to run the project.

---