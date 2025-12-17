# Automated Analysis Bot

An interactive web application built with Streamlit that allows users to upload a dataset, perform comprehensive data preprocessing, and generate a wide range of exploratory data analysis (EDA) visualizations and reports.

##  Features

### 1. Data Loading
- **Upload Various Formats**: Load datasets from CSV, Excel (`.xlsx`, `.xls`), and JSON files.
- **Instant Feedback**: Get immediate information on dataset shape, column names, and data types upon upload.

### 2. Data Preprocessing
A complete suite of tools to clean and prepare your data for analysis.
- **Drop Unnecessary Columns**: Select and remove irrelevant columns.
- **Handle Missing Values**:
    - Target specific columns with missing data.
    - Choose from multiple imputation strategies (Mean, Median, Mode) or fill with a specific value.
    - Option to drop rows with missing values.
- **Remove Duplicate Rows**: Identify and remove duplicate entries from the dataset.
- **Feature Scaling**:
    - Apply `StandardScaler` or `MinMaxScaler` to selected numeric columns to normalize their range.

### 3. Data Analysis
A powerful set of tools to explore and visualize your cleaned data.
- **Full Insights Report**:
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
    - Automatically detects date/time columns.
    - Plot any numeric value over time in an interactive line chart.

##  How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayushajmera/visualization-bot.git
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

## File Structure

The project is organized into a modular structure for clarity and maintainability:

- **`streamlit_app.py`**: The main entry point of the application. It handles the user interface (UI) and the overall workflow.
- **`backend.py`**: A reusable "toolbox" containing all the core, low-level functions for data manipulation (e.g., `load_dataset`), plotting (e.g., `plot_histogram`), and calculations (e.g., `scale_features`).
- **`full_report.py`**: The "report manager" that orchestrates the creation of the "Full Insights Report" by calling the necessary functions from `backend.py` in a specific sequence.
- **`requirements.txt`**: A list of all Python libraries required to run the project.

---
