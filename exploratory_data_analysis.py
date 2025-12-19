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

    for idx, col in enumerate(df.columns):
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
                hist_res = be.plot_histogram(df, col)
                if hist_res is None:
                    st.info("No histogram available for this column.")
                elif isinstance(hist_res, tuple) and len(hist_res) == 2:
                    hist_fig, hist_buf = hist_res
                    st.pyplot(hist_fig)
                    if hist_buf:
                        st.download_button("Download Histogram", hist_buf, f"histogram_{col}.png", "image/png")
                    try:
                        plt.close(hist_fig)
                    except Exception:
                        pass
                else:
                    # Assume Plotly Figure
                    hist_fig = hist_res
                    st.plotly_chart(hist_fig, use_container_width=True)
                    try:
                        img_bytes = hist_fig.to_image(format="png")
                        st.download_button("Download Histogram", img_bytes, f"histogram_{col}.png", "image/png")
                    except Exception:
                        st.info("PNG download unavailable (install 'kaleido' to enable).")

                box_res = be.plot_box_plot(df, col)
                if box_res is None:
                    st.info("No box plot available for this column.")
                elif isinstance(box_res, tuple) and len(box_res) == 2:
                    box_fig, box_buf = box_res
                    st.pyplot(box_fig)
                    if box_buf:
                        st.download_button("Download Box Plot", box_buf, f"boxplot_{col}.png", "image/png")
                    try:
                        plt.close(box_fig)
                    except Exception:
                        pass
                else:
                    box_fig = box_res
                    st.plotly_chart(box_fig, use_container_width=True)
                    try:
                        img_bytes = box_fig.to_image(format="png")
                        st.download_button("Download Box Plot", img_bytes, f"boxplot_{col}.png", "image/png")
                    except Exception:
                        st.info("PNG download unavailable (install 'kaleido' to enable).")

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
                    # Allow the user to control how many categories to show in the bar chart. Categories beyond this are grouped into 'Other'.
                    max_bars = st.slider(
                        "Max bars (top N categories; others grouped into 'Other')",
                        min_value=2, max_value=50, value=20, step=1,
                        help="Limit the number of bars so the chart remains readable.",
                        key=f"max_bar_slices_{idx}_{col}"
                    )
                    bar_res = be.plot_bar_chart(df, col, max_bars=max_bars)
                    if bar_res is None:
                        st.info("No bar chart available for this column.")
                    elif isinstance(bar_res, tuple) and len(bar_res) == 2:
                        bar_fig, bar_buf = bar_res
                        st.pyplot(bar_fig)
                        if bar_buf:
                            st.download_button("Download Bar Chart", bar_buf, f"barchart_{col}.png", "image/png")
                        try:
                            plt.close(bar_fig)
                        except Exception:
                            pass
                    else:
                        bar_fig = bar_res
                        st.plotly_chart(bar_fig, use_container_width=True)
                        try:
                            img_bytes = bar_fig.to_image(format="png")
                            st.download_button("Download Bar Chart", img_bytes, f"barchart_{col}.png", "image/png")
                        except Exception:
                            st.info("PNG download unavailable (install 'kaleido' to enable).")

                    st.markdown("###### Pie Chart")
                    # Allow the user to control how many slices to show in the pie chart. Categories beyond this are grouped into 'Other'.
                    max_slices = st.slider(
                        "Max pie slices (top N categories; others grouped into 'Other')",
                        min_value=2, max_value=30, value=10, step=1,
                        help="Limit the number of slices so the pie remains readable.",
                        key=f"max_pie_slices_{idx}_{col}"
                    )
                    pie_res = be.plot_pie_chart(df, col, max_slices=max_slices)
                    if pie_res is None:
                        st.info("No pie chart available for this column.")
                    elif isinstance(pie_res, tuple) and len(pie_res) == 2:
                        pie_fig, pie_buf = pie_res
                        st.pyplot(pie_fig)
                        if pie_buf:
                            st.download_button("Download Pie Chart", pie_buf, f"piechart_{col}.png", "image/png")
                        try:
                            plt.close(pie_fig)
                        except Exception:
                            pass
                    else:
                        pie_fig = pie_res
                        st.plotly_chart(pie_fig, use_container_width=True)
                        try:
                            img_bytes = pie_fig.to_image(format="png")
                            st.download_button("Download Pie Chart", img_bytes, f"piechart_{col}.png", "image/png")
                        except Exception:
                            st.info("PNG download unavailable (install 'kaleido' to enable).")
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
        heatmap_res = be.plot_correlation_heatmap(df)
        if heatmap_res is None:
            st.info("No correlation heatmap available for this dataset.")
        elif isinstance(heatmap_res, tuple) and len(heatmap_res) == 2:
            heatmap_fig, heatmap_buf = heatmap_res
            st.pyplot(heatmap_fig)
            if heatmap_buf:
                st.download_button("Download Heatmap", heatmap_buf, "correlation_heatmap.png", "image/png")
            try:
                plt.close(heatmap_fig)
            except Exception:
                pass
        else:
            heatmap_fig = heatmap_res
            st.plotly_chart(heatmap_fig, use_container_width=True)
            try:
                img_bytes = heatmap_fig.to_image(format="png")
                st.download_button("Download Heatmap", img_bytes, "correlation_heatmap.png", "image/png")
            except Exception:
                st.info("PNG download unavailable (install 'kaleido' to enable).")

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
                    # Create a unique key for each scatter chart to avoid Streamlit DuplicateElementId errors
                    safe_var1 = str(var1).replace(' ', '_').replace('.', '_')
                    safe_var2 = str(var2).replace(' ', '_').replace('.', '_')
                    st.plotly_chart(
                        scatter_fig,
                        width='stretch',
                        key=f"scatter_{safe_var1}__{safe_var2}"
                    )

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
                    cat_box_res = be.plot_categorical_boxplot(df, cat_col, num_col)
                    if cat_box_res is None:
                        st.info("No categorical boxplot available for this pair.")
                    elif isinstance(cat_box_res, tuple) and len(cat_box_res) == 2:
                        cat_box_fig, cat_box_buf = cat_box_res
                        st.pyplot(cat_box_fig)
                        if cat_box_buf:
                            st.download_button(f"Download {num_col} vs {cat_col} Plot", cat_box_buf, f"cat_boxplot_{num_col}_{cat_col}.png", "image/png")
                        try:
                            plt.close(cat_box_fig)
                        except Exception:
                            pass
                    else:
                        cat_box_fig = cat_box_res
                        st.plotly_chart(cat_box_fig, use_container_width=True)
                        try:
                            img_bytes = cat_box_fig.to_image(format="png")
                            st.download_button(f"Download {num_col} vs {cat_col} Plot", img_bytes, f"cat_boxplot_{num_col}_{cat_col}.png", "image/png")
                        except Exception:
                            st.info("PNG download unavailable (install 'kaleido' to enable).")
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