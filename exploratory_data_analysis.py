# exploratory_data_analysis.py
# Handles the "Exploratory Data Analysis (EDA)" feature.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def perform_eda(df, be):
    """
    Performs and displays a comprehensive Exploratory Data Analysis.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze.
        be (module): The backend module containing helper functions.
    """
    st.header("Exploratory Data Analysis")

    # --- 5.1 Univariate Analysis ---
    st.subheader("Univariate Analysis (Single Variable)")
    st.markdown("Here we analyze each column individually to understand its characteristics.")

    for col in df.columns:
        with st.expander(f"Analysis for Column: `{col}`"):
            if pd.api.types.is_numeric_dtype(df[col]):
                st.markdown("#### Numerical Column Analysis")
                
                # --- Basic Stats ---
                col1,col2,col3,col4 = st.columns(4)
                col1.metric("Mean", f"{df[col].mean():.2f}")
                col2.metric("Median", f"{df[col].median():.2f}")
                col3.metric("Min", f"{df[col].min():.2f}")
                col4.metric("Max", f"{df[col].max():.2f}")

                # --- Distribution and Outliers ---
                st.markdown("##### Distribution Shape & Outliers")
                hist_fig, hist_buf = be.plot_histogram(df, col)
                # Display the generated image buffer for consistent rendering
                st.image(hist_buf, use_column_width=True)
                st.download_button("Download Histogram", hist_buf, f"histogram_{col}.png", "image/png")

                box_fig, box_buf = be.plot_box_plot(df, col)
                # Display the generated image buffer for consistent rendering
                st.image(box_buf, use_column_width=True)
                st.download_button("Download Box Plot", box_buf, f"boxplot_{col}.png", "image/png")

            else: # Categorical
                st.markdown("#### Categorical Column Analysis")
                
                # --- Frequency Counts ---
                st.markdown("Frequency Counts")
                counts = df[col].value_counts().nlargest(20) # Limit to top 20 for readability
                
                if not counts.empty:
                    st.write(f"**Most Common Category:** `{counts.index[0]}` ({counts.iloc[0]} occurrences)")
                    st.dataframe(counts.to_frame("Count"))
                    
                    # --- Visualization ---
                    st.markdown("Visualizations")
                    
                    st.markdown("###### Bar Chart")
                    bar_fig, bar_buf = be.plot_bar_chart(df, col)
                    # Use the image buffer to display the chart reliably in Streamlit
                    st.image(bar_buf, use_column_width=True)
                    st.download_button("Download Bar Chart", bar_buf, f"barchart_{col}.png", "image/png")

                    st.markdown("###### Pie Chart")
                    pie_fig, pie_buf = be.plot_pie_chart(df, col)
                    # Use the image buffer to display the chart reliably in Streamlit
                    st.image(pie_buf, use_column_width=True)
                    st.download_button("Download Pie Chart", pie_buf, f"piechart_{col}.png", "image/png")
                else:
                    st.info("This column contains no data to analyze.")

    # --- 5.2 Bivariate Analysis ---
    st.subheader("Bivariate Analysis (Two Variables)")
    st.markdown("Now, let's explore the relationships between pairs of variables.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Numeric vs. Numeric ---
    with st.expander("Numeric vs. Numeric Analysis"):
        st.markdown("#### Correlation Heatmap")
        st.write("The heatmap below shows the linear correlation between all numeric variables. Values close to 1 (red) or -1 (blue) indicate a strong relationship.")
        heatmap_fig, heatmap_buf = be.plot_correlation_heatmap(df)
        if heatmap_fig:
            # Use the saved image buffer to display the heatmap reliably
            st.image(heatmap_buf, use_column_width=True)
            st.download_button("Download Heatmap", heatmap_buf, "correlation_heatmap.png", "image/png")

        if len(numeric_cols) >= 2:
            st.markdown("#### Most Correlated Variable Pairs")
            st.write("Here are scatter plots for the variable pairs with the strongest correlations.")
            corr_matrix = df[numeric_cols].corr().abs()
            # Unstack the matrix to get pairs, and remove self-correlations
            sol = corr_matrix.unstack()
            so = sol.sort_values(kind="quicksort", ascending=False)
            # Filter out self-correlations and duplicates
            so = so[so < 1]
            top_pairs = so.drop_duplicates().head(3)

            for (var1, var2), corr_val in top_pairs.items():
                st.write(f"**`{var1}`** vs **`{var2}`** (Correlation: {df[var1].corr(df[var2]):.2f})")
                # Call the cached function and then render the UI element
                scatter_fig = be.plot_scatter_plot(df, var1, var2)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)

    # --- Numeric vs. Categorical ---
    with st.expander("Numeric vs. Categorical Analysis"):
        st.write("This section shows how numeric distributions differ across categories.")
        if categorical_cols and numeric_cols:
            # Select a few representative pairs to avoid overwhelming the user
            cat_to_plot = [c for c in categorical_cols if df[c].nunique() < 15] # Only plot for categoricals with fewer than 15 unique values
            
            if not cat_to_plot:
                st.info("No categorical columns with a reasonable number of unique values (<15) to display.")
            else:
                # For simplicity, let's pick the first categorical and first few numeric columns
                cat_col = cat_to_plot[0]
                st.write(f"Showing distribution of numeric variables across categories of **`{cat_col}`**.")
                for num_col in numeric_cols[:3]: # Limit to first 3 numeric cols for brevity
                    st.markdown(f"##### `{num_col}` across `{cat_col}` categories")
                    cat_box_fig, cat_box_buf = be.plot_categorical_boxplot(df, cat_col, num_col)
                    if cat_box_fig:
                        # Display via buffer for consistent rendering
                        st.image(cat_box_buf, use_column_width=True)
                        st.download_button(f"Download {num_col} vs {cat_col} Plot", cat_box_buf, f"cat_boxplot_{num_col}_{cat_col}.png", "image/png")
        else:
            st.info("Not enough numeric and categorical columns to perform this analysis.")

    # --- 5.3 Multivariate Analysis ---
    st.subheader("Multivariate Analysis (Multiple Variables)")
    st.markdown("Finally, let's look for patterns and interactions across many variables at once.")
    
    with st.expander("Grouped Analysis (Numeric by 2 Categories)"):
        be.plot_grouped_analysis(df)

    with st.expander("Faceted Scatter Analysis (2 Numerics by 1 Category)"):
        be.plot_faceted_scatter(df)

    # --- 5.4 Segmentation Analysis ---
    # Segmentation & Grouping removed from the UI because it's not functional.
    # If needed later, the backend implementation (`perform_segmentation_analysis`) remains available and
    # can be restored after a refactor that fixes the identified issues (e.g., visualization bugs, caching side effects).
    # be.perform_segmentation_analysis(df)  # Removed from UI (kept as reference)