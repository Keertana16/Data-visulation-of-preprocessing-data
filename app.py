import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Heart Disease Data Visualization", layout="wide")
st.title("📊 Heart Disease Data Visualization Dashboard")
st.markdown("*Displaying preprocessed data with advanced visualizations*")
st.markdown("---")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    try:
        # Load data with absolute path
        import os
        csv_path = os.path.join(os.path.dirname(__file__), 'heart.csv')
        data = pd.read_csv(csv_path)
        data = data.dropna()
        
        # Data cleaning - Replace zero values with median
        data["Cholesterol"] = data["Cholesterol"].replace(0, data["Cholesterol"].median())
        data["RestingBP"] = data["RestingBP"].replace(0, data["RestingBP"].median())
        data["Oldpeak"] = data["Oldpeak"].fillna(data["Oldpeak"].median())
        
        # Handle outliers using IQR method
        def cap_outliers(col):
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower, upper)
        
        for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
            cap_outliers(col)
        
        # Log transform skewed features
        data["Oldpeak_log"] = np.log1p(data["Oldpeak"])
        data["Cholesterol_log"] = np.log1p(data["Cholesterol"])
        
        # Final check - remove any remaining NaN values
        data = data.dropna()
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
data = load_and_preprocess_data()

if data is not None:
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    st.sidebar.header("📈 Visualization Settings")
    
    # Select visualization library
    library = st.sidebar.selectbox(
        "Select Visualization Library:",
        ["Matplotlib", "Seaborn", "Plotly"]
    )
    
    # Get available columns (numeric only for plotting)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    # Define graph types for each library
    graph_types = {
        "Matplotlib": [
            "Line Plot", "Multi-Line Plot", "Bar Chart", "Histogram",
            "Scatter Plot", "Box Plot", "Area Plot", "Pie Chart",
            "Hexbin Plot", "Contour Plot", "3D Scatter Plot",
            "3D Surface Plot", "Polar Plot", "Quiver Plot (Arrow Plot)",
            "Stream Plot"
        ],
        "Seaborn": [
            "Line Plot", "Bar Plot", "Count Plot", "Histogram", "KDE Plot",
            "Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot",
            "Regression Plot", "Pair Plot", "Joint Plot", "Heatmap",
            "Clustermap"
        ],
        "Plotly": [
            "Interactive Line Chart", "Interactive Bar Chart",
            "Interactive Scatter Plot", "Bubble Chart", "Histogram",
            "Box Plot", "Violin Plot", "Area Chart", "3D Scatter Plot",
            "3D Surface Plot", "Treemap", "Sunburst Chart",
            "Funnel Chart", "Waterfall Chart", "Sankey Diagram",
            "Parallel Coordinates Plot", "Radar Chart", "Gauge Chart"
        ]
    }
    
    # Select graph type
    graph_type = st.sidebar.selectbox(
        "Select Graph Type:",
        graph_types[library]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Columns")
    
    # ========================================================================
    # MATPLOTLIB FUNCTIONS
    # ========================================================================
    
    def matplotlib_line_plot():
        """Create a line plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="line_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="line_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data[x_col], data[y_col], marker='o', linestyle='-', linewidth=2)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"{x_col} vs {y_col} - Line Plot", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_multi_line_plot():
        """Create a multi-line plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="mline_x")
        y_cols = st.sidebar.multiselect("Y-axis (select multiple):", numeric_cols, default=[numeric_cols[1], numeric_cols[2]] if len(numeric_cols) > 2 else numeric_cols)
        
        if not y_cols:
            st.warning("Please select at least one Y column")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in y_cols:
                ax.plot(data[x_col], data[col], marker='o', label=col, linewidth=2)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.set_title(f"Multi-Line Plot - Multiple Columns vs {x_col}", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_bar_chart():
        """Create a bar chart using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="bar_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="bar_y")
        
        try:
            if x_col in categorical_cols:
                grouped_data = data.groupby(x_col)[y_col].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(grouped_data[x_col].astype(str), grouped_data[y_col], color='steelblue', edgecolor='black')
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(data)), data[y_col], color='steelblue', edgecolor='black')
            
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"{x_col} vs {y_col} - Bar Chart", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_histogram():
        """Create a histogram using Matplotlib"""
        col = st.sidebar.selectbox("Column:", numeric_cols, key="hist_col")
        bins = st.sidebar.slider("Number of bins:", 10, 50, 30)
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data[col], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"Distribution of {col} - Histogram", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_scatter_plot():
        """Create a scatter plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="scatter_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="scatter_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(data[x_col], data[y_col], alpha=0.6, s=50, color='steelblue', edgecolor='black')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"{x_col} vs {y_col} - Scatter Plot", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_box_plot():
        """Create a box plot using Matplotlib"""
        col = st.sidebar.selectbox("Column:", numeric_cols, key="box_col")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data[col], vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue'),
                      medianprops=dict(color='red', linewidth=2))
            ax.set_ylabel(col, fontsize=12)
            ax.set_title(f"Box Plot of {col}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_area_plot():
        """Create an area plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="area_x")
        y_cols = st.sidebar.multiselect("Y-axis (select multiple):", numeric_cols, default=[numeric_cols[1], numeric_cols[2]] if len(numeric_cols) > 2 else numeric_cols, key="area_y")
        
        if not y_cols:
            st.warning("Please select at least one Y column")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_data = data.sort_values(x_col)
            for col in y_cols:
                ax.fill_between(sorted_data[x_col], sorted_data[col], alpha=0.5, label=col)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.set_title(f"Area Plot - {x_col} vs Multiple Columns", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_pie_chart():
        """Create a pie chart using Matplotlib"""
        col = st.sidebar.selectbox("Column (will group by value counts):", all_cols, key="pie_col")
        
        try:
            if col in categorical_cols:
                data_counts = data[col].value_counts()
            else:
                data_counts = data[col].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Pie Chart of {col}", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_hexbin_plot():
        """Create a hexbin plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="hex_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="hex_y")
        gridsize = st.sidebar.slider("Grid size:", 5, 30, 15)
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            hexbin = ax.hexbin(data[x_col], data[y_col], gridsize=gridsize, cmap='YlOrRd', mincnt=1)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"{x_col} vs {y_col} - Hexbin Plot", fontsize=14, fontweight='bold')
            plt.colorbar(hexbin, ax=ax, label='Count')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_contour_plot():
        """Create a contour plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="contour_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="contour_y")
        
        try:
            # Create grid for contour
            x = data[x_col].values
            y = data[y_col].values
            
            # Create meshgrid
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            # Calculate Z values (density)
            from scipy.stats import gaussian_kde
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = gaussian_kde(np.vstack([x, y]))(positions).reshape(xx.shape)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            contour = ax.contourf(xx, yy, z, levels=15, cmap='viridis')
            ax.scatter(x, y, c='red', s=20, alpha=0.5)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"{x_col} vs {y_col} - Contour Plot", fontsize=14, fontweight='bold')
            plt.colorbar(contour, ax=ax, label='Density')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_3d_scatter():
        """Create a 3D scatter plot using Matplotlib"""
        from mpl_toolkits.mplot3d import Axes3D
        
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="3d_scatter_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="3d_scatter_y")
        z_col = st.sidebar.selectbox("Z-axis:", numeric_cols, key="3d_scatter_z")
        
        try:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[x_col], data[y_col], data[z_col], c='steelblue', marker='o', s=50, alpha=0.6)
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(y_col, fontsize=10)
            ax.set_zlabel(z_col, fontsize=10)
            ax.set_title(f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}", fontsize=12, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_3d_surface():
        """Create a 3D surface plot using Matplotlib"""
        from mpl_toolkits.mplot3d import Axes3D
        
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="3d_surf_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="3d_surf_y")
        
        try:
            x = data[x_col].values
            y = data[y_col].values
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50))
            
            from scipy.stats import gaussian_kde
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = gaussian_kde(np.vstack([x, y]))(positions).reshape(xx.shape)
            
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.8)
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(y_col, fontsize=10)
            ax.set_zlabel("Density", fontsize=10)
            ax.set_title(f"3D Surface Plot", fontsize=12, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_polar_plot():
        """Create a polar plot using Matplotlib"""
        col = st.sidebar.selectbox("Column (for values):", numeric_cols, key="polar_col")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            theta = np.linspace(0, 2*np.pi, len(data))
            r = (data[col] - data[col].min()) / (data[col].max() - data[col].min())  # Normalize
            ax.scatter(theta, r, alpha=0.6, s=50)
            ax.set_title(f"Polar Plot of {col}", fontsize=14, fontweight='bold', pad=20)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_quiver_plot():
        """Create a quiver (arrow) plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X position:", numeric_cols, key="quiver_x")
        y_col = st.sidebar.selectbox("Y position:", numeric_cols, key="quiver_y")
        u_col = st.sidebar.selectbox("X direction:", numeric_cols, key="quiver_u")
        v_col = st.sidebar.selectbox("Y direction:", numeric_cols, key="quiver_v")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sample = data.sample(n=min(50, len(data)))  # Sample for clarity
            ax.quiver(sample[x_col], sample[y_col], sample[u_col], sample[v_col], alpha=0.6)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"Quiver Plot (Arrow Plot)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def matplotlib_stream_plot():
        """Create a stream plot using Matplotlib"""
        x_col = st.sidebar.selectbox("X position:", numeric_cols, key="stream_x")
        y_col = st.sidebar.selectbox("Y position:", numeric_cols, key="stream_y")
        
        try:
            x = data[x_col].values
            y = data[y_col].values
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 20),
                              np.linspace(y_min, y_max, 20))
            
            # Create synthetic velocity field
            U = np.sin(X / (x_max - x_min) * 4 * np.pi)
            V = np.cos(Y / (y_max - y_min) * 4 * np.pi)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.streamplot(X[0, :], Y[:, 0], U, V, color='steelblue', density=1.5)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f"Stream Plot", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    # ========================================================================
    # SEABORN FUNCTIONS
    # ========================================================================
    
    def seaborn_line_plot():
        """Create a line plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="sns_line_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="sns_line_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data, x=x_col, y=y_col, ax=ax, marker='o')
            ax.set_title(f"{x_col} vs {y_col} - Line Plot", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_bar_plot():
        """Create a bar plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="sns_bar_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="sns_bar_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax, palette='Set2')
            ax.set_title(f"{x_col} vs {y_col} - Bar Plot", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_count_plot():
        """Create a count plot using Seaborn"""
        col = st.sidebar.selectbox("Column:", all_cols, key="sns_count_col")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=data, x=col, ax=ax, palette='Set2')
            ax.set_title(f"Count Plot of {col}", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_histogram():
        """Create a histogram using Seaborn"""
        col = st.sidebar.selectbox("Column:", numeric_cols, key="sns_hist_col")
        bins = st.sidebar.slider("Number of bins:", 10, 50, 30, key="sns_hist_bins")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=data, x=col, bins=bins, ax=ax, kde=True, palette='Set2')
            ax.set_title(f"Distribution of {col} - Histogram", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_kde_plot():
        """Create a KDE plot using Seaborn"""
        col = st.sidebar.selectbox("Column:", numeric_cols, key="sns_kde_col")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(data=data, x=col, ax=ax, fill=True, palette='Set2')
            ax.set_title(f"KDE Plot of {col}", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_box_plot():
        """Create a box plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="sns_box_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="sns_box_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data, x=x_col, y=y_col, ax=ax, palette='Set2')
            ax.set_title(f"{x_col} vs {y_col} - Box Plot", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_violin_plot():
        """Create a violin plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="sns_violin_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="sns_violin_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=data, x=x_col, y=y_col, ax=ax, palette='Set2')
            ax.set_title(f"{x_col} vs {y_col} - Violin Plot", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_strip_plot():
        """Create a strip plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="sns_strip_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="sns_strip_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.stripplot(data=data, x=x_col, y=y_col, ax=ax, palette='Set2', alpha=0.6, size=6)
            ax.set_title(f"{x_col} vs {y_col} - Strip Plot", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_swarm_plot():
        """Create a swarm plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="sns_swarm_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="sns_swarm_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.swarmplot(data=data, x=x_col, y=y_col, ax=ax, palette='Set2', size=6)
            ax.set_title(f"{x_col} vs {y_col} - Swarm Plot", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_regression_plot():
        """Create a regression plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="sns_reg_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="sns_reg_y")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(data=data, x=x_col, y=y_col, ax=ax, scatter_kws={'alpha': 0.6})
            ax.set_title(f"{x_col} vs {y_col} - Regression Plot", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_pair_plot():
        """Create a pair plot using Seaborn"""
        cols = st.sidebar.multiselect("Columns (select 2-5):", numeric_cols, default=numeric_cols[:3], key="sns_pair_cols")
        
        if not cols or len(cols) < 2:
            st.warning("Please select at least 2 columns")
            return
        
        try:
            plot = sns.pairplot(data[cols], diag_kind='kde', plot_kws={'alpha': 0.6})
            st.pyplot(plot.fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_joint_plot():
        """Create a joint plot using Seaborn"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="sns_joint_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="sns_joint_y")
        kind = st.sidebar.selectbox("Plot kind:", ["scatter", "kde", "hist", "hex"], key="sns_joint_kind")
        
        try:
            plot = sns.jointplot(data=data, x=x_col, y=y_col, kind=kind)
            st.pyplot(plot.fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_heatmap():
        """Create a heatmap using Seaborn"""
        cols = st.sidebar.multiselect("Columns (numeric):", numeric_cols, default=numeric_cols[:5], key="sns_heat_cols")
        
        if not cols:
            st.warning("Please select at least 1 column")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_matrix = data[cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0)
            ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def seaborn_clustermap():
        """Create a clustermap using Seaborn"""
        cols = st.sidebar.multiselect("Columns (numeric):", numeric_cols, default=numeric_cols[:5], key="sns_cluster_cols")
        
        if not cols:
            st.warning("Please select at least 1 column")
            return
        
        try:
            plot = sns.clustermap(data[cols].corr(), cmap='coolwarm', center=0, figsize=(8, 6))
            st.pyplot(plot.fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    # ========================================================================
    # PLOTLY FUNCTIONS
    # ========================================================================
    
    def plotly_line_chart():
        """Create an interactive line chart using Plotly"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="ply_line_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="ply_line_y")
        
        try:
            fig = px.line(data, x=x_col, y=y_col, markers=True,
                         title=f"{x_col} vs {y_col} - Interactive Line Chart",
                         labels={x_col: x_col, y_col: y_col})
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_bar_chart():
        """Create an interactive bar chart using Plotly"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="ply_bar_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="ply_bar_y")
        
        try:
            if x_col in categorical_cols:
                grouped_data = data.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(grouped_data, x=x_col, y=y_col,
                           title=f"{x_col} vs {y_col} - Interactive Bar Chart",
                           labels={x_col: x_col, y_col: y_col})
            else:
                fig = px.bar(data, x=x_col, y=y_col,
                           title=f"{x_col} vs {y_col} - Interactive Bar Chart",
                           labels={x_col: x_col, y_col: y_col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_scatter_plot():
        """Create an interactive scatter plot using Plotly"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="ply_scatter_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="ply_scatter_y")
        
        try:
            fig = px.scatter(data, x=x_col, y=y_col,
                           title=f"{x_col} vs {y_col} - Interactive Scatter Plot",
                           labels={x_col: x_col, y_col: y_col})
            fig.update_traces(marker=dict(size=8, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_bubble_chart():
        """Create a bubble chart using Plotly"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="ply_bubble_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="ply_bubble_y")
        size_col = st.sidebar.selectbox("Bubble Size:", numeric_cols, key="ply_bubble_size")
        
        try:
            fig = px.scatter(data, x=x_col, y=y_col, size=size_col,
                           title=f"Bubble Chart: {x_col} vs {y_col}",
                           labels={x_col: x_col, y_col: y_col, size_col: size_col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_histogram():
        """Create an interactive histogram using Plotly"""
        col = st.sidebar.selectbox("Column:", numeric_cols, key="ply_hist_col")
        bins = st.sidebar.slider("Number of bins:", 10, 50, 30, key="ply_hist_bins")
        
        try:
            fig = px.histogram(data, x=col, nbins=bins,
                            title=f"Distribution of {col} - Interactive Histogram",
                            labels={col: col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_box_plot():
        """Create an interactive box plot using Plotly"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="ply_box_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="ply_box_y")
        
        try:
            fig = px.box(data, x=x_col, y=y_col,
                       title=f"{x_col} vs {y_col} - Interactive Box Plot",
                       labels={x_col: x_col, y_col: y_col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_violin_plot():
        """Create an interactive violin plot using Plotly"""
        x_col = st.sidebar.selectbox("X-axis (categorical):", all_cols, key="ply_violin_x")
        y_col = st.sidebar.selectbox("Y-axis (numeric):", numeric_cols, key="ply_violin_y")
        
        try:
            fig = px.violin(data, x=x_col, y=y_col,
                          title=f"{x_col} vs {y_col} - Interactive Violin Plot",
                          labels={x_col: x_col, y_col: y_col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_area_chart():
        """Create an interactive area chart using Plotly"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="ply_area_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="ply_area_y")
        
        try:
            sorted_data = data.sort_values(x_col)
            fig = px.area(sorted_data, x=x_col, y=y_col,
                        title=f"Interactive Area Chart: {x_col} vs {y_col}",
                        labels={x_col: x_col, y_col: y_col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_3d_scatter():
        """Create an interactive 3D scatter plot using Plotly"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="ply_3d_scatter_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="ply_3d_scatter_y")
        z_col = st.sidebar.selectbox("Z-axis:", numeric_cols, key="ply_3d_scatter_z")
        
        try:
            fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                              title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
                              labels={x_col: x_col, y_col: y_col, z_col: z_col})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_3d_surface():
        """Create an interactive 3D surface plot using Plotly"""
        x_col = st.sidebar.selectbox("X-axis:", numeric_cols, key="ply_3d_surf_x")
        y_col = st.sidebar.selectbox("Y-axis:", numeric_cols, key="ply_3d_surf_y")
        
        try:
            x = data[x_col].values
            y = data[y_col].values
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 50),
                              np.linspace(y_min, y_max, 50))
            
            from scipy.stats import gaussian_kde
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = gaussian_kde(np.vstack([x, y]))(positions).reshape(X.shape)
            
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
            fig.update_layout(title="3D Surface Plot", scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title="Density"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_treemap():
        """Create a treemap using Plotly"""
        col = st.sidebar.selectbox("Column (will group by value counts):", categorical_cols if categorical_cols else numeric_cols, key="ply_tree_col")
        
        try:
            if col in categorical_cols:
                counts = data[col].value_counts().reset_index()
                counts.columns = [col, 'count']
            else:
                counts = data[col].value_counts().head(10).reset_index()
                counts.columns = [col, 'count']
            
            fig = px.treemap(counts, labels=col, values='count',
                           title=f"Treemap of {col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_sunburst():
        """Create a sunburst chart using Plotly"""
        col = st.sidebar.selectbox("Column (will group by value counts):", categorical_cols if categorical_cols else numeric_cols, key="ply_sun_col")
        
        try:
            if col in categorical_cols:
                counts = data[col].value_counts().reset_index()
                counts.columns = [col, 'count']
            else:
                counts = data[col].value_counts().head(10).reset_index()
                counts.columns = [col, 'count']
            
            fig = px.sunburst(counts, labels=col, values='count',
                            title=f"Sunburst Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_funnel_chart():
        """Create a funnel chart using Plotly"""
        col = st.sidebar.selectbox("Column (will show top 10):", categorical_cols if categorical_cols else numeric_cols, key="ply_funnel_col")
        
        try:
            if col in categorical_cols:
                counts = data[col].value_counts().reset_index().head(10)
                counts.columns = [col, 'count']
            else:
                counts = data[col].value_counts().head(10).reset_index()
                counts.columns = [col, 'count']
            
            fig = px.funnel(counts, x='count', y=col,
                          title=f"Funnel Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_waterfall_chart():
        """Create a waterfall chart using Plotly"""
        col = st.sidebar.selectbox("Column (will show top 5):", numeric_cols, key="ply_waterfall_col")
        
        try:
            sorted_data = data.nlargest(5, col)
            fig = go.Figure(go.Waterfall(
                x=range(len(sorted_data)),
                y=sorted_data[col],
                text=sorted_data.index.astype(str),
                textposition="outside",
                connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            ))
            fig.update_layout(title=f"Waterfall Chart of {col}", xaxis_title="Index", yaxis_title=col)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_sankey_diagram():
        """Create a Sankey diagram using Plotly"""
        source_col = st.sidebar.selectbox("Source Column:", all_cols, key="ply_sankey_src")
        target_col = st.sidebar.selectbox("Target Column:", all_cols, key="ply_sankey_tgt")
        
        try:
            flow_data = data[[source_col, target_col]].astype(str).value_counts().reset_index()
            flow_data.columns = ['source', 'target', 'value']
            
            source_idx = {label: idx for idx, label in enumerate(pd.concat([flow_data['source'], flow_data['target']]).unique())}
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=list(source_idx.keys())),
                link=dict(
                    source=[source_idx[x] for x in flow_data['source']],
                    target=[source_idx[x] for x in flow_data['target']],
                    value=flow_data['value']
                )
            )])
            fig.update_layout(title="Sankey Diagram", font=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_parallel_coordinates():
        """Create a parallel coordinates plot using Plotly"""
        cols = st.sidebar.multiselect("Columns (numeric):", numeric_cols, default=numeric_cols[:3], key="ply_parallel_cols")
        
        if not cols:
            st.warning("Please select at least 1 column")
            return
        
        try:
            fig = px.parallel_coordinates(data[cols],
                                        title="Parallel Coordinates Plot",
                                        color=data[cols[0]],
                                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_radar_chart():
        """Create a radar chart using Plotly"""
        cols = st.sidebar.multiselect("Columns (numeric, max 6):", numeric_cols[:6], default=numeric_cols[:4], key="ply_radar_cols")
        
        if not cols:
            st.warning("Please select at least 1 column")
            return
        
        try:
            means = data[cols].mean()
            fig = go.Figure(data=go.Scatterpolar(
                r=means.values,
                theta=means.index,
                fill='toself'
            ))
            fig.update_layout(title="Radar Chart (Mean Values)", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    def plotly_gauge_chart():
        """Create a gauge chart using Plotly"""
        col = st.sidebar.selectbox("Column:", numeric_cols, key="ply_gauge_col")
        
        try:
            value = data[col].mean()
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                title={'text': f"Gauge: Average {col}"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [data[col].min(), data[col].max()]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [data[col].min(), data[col].quantile(0.33)], 'color': "lightgray"},
                        {'range': [data[col].quantile(0.33), data[col].quantile(0.66)], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': data[col].quantile(0.75)
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    # ========================================================================
    # MAIN VISUALIZATION LOGIC
    # ========================================================================
    
    # Define function mapping
    function_map = {
        "Matplotlib": {
            "Line Plot": matplotlib_line_plot,
            "Multi-Line Plot": matplotlib_multi_line_plot,
            "Bar Chart": matplotlib_bar_chart,
            "Histogram": matplotlib_histogram,
            "Scatter Plot": matplotlib_scatter_plot,
            "Box Plot": matplotlib_box_plot,
            "Area Plot": matplotlib_area_plot,
            "Pie Chart": matplotlib_pie_chart,
            "Hexbin Plot": matplotlib_hexbin_plot,
            "Contour Plot": matplotlib_contour_plot,
            "3D Scatter Plot": matplotlib_3d_scatter,
            "3D Surface Plot": matplotlib_3d_surface,
            "Polar Plot": matplotlib_polar_plot,
            "Quiver Plot (Arrow Plot)": matplotlib_quiver_plot,
            "Stream Plot": matplotlib_stream_plot,
        },
        "Seaborn": {
            "Line Plot": seaborn_line_plot,
            "Bar Plot": seaborn_bar_plot,
            "Count Plot": seaborn_count_plot,
            "Histogram": seaborn_histogram,
            "KDE Plot": seaborn_kde_plot,
            "Box Plot": seaborn_box_plot,
            "Violin Plot": seaborn_violin_plot,
            "Strip Plot": seaborn_strip_plot,
            "Swarm Plot": seaborn_swarm_plot,
            "Regression Plot": seaborn_regression_plot,
            "Pair Plot": seaborn_pair_plot,
            "Joint Plot": seaborn_joint_plot,
            "Heatmap": seaborn_heatmap,
            "Clustermap": seaborn_clustermap,
        },
        "Plotly": {
            "Interactive Line Chart": plotly_line_chart,
            "Interactive Bar Chart": plotly_bar_chart,
            "Interactive Scatter Plot": plotly_scatter_plot,
            "Bubble Chart": plotly_bubble_chart,
            "Histogram": plotly_histogram,
            "Box Plot": plotly_box_plot,
            "Violin Plot": plotly_violin_plot,
            "Area Chart": plotly_area_chart,
            "3D Scatter Plot": plotly_3d_scatter,
            "3D Surface Plot": plotly_3d_surface,
            "Treemap": plotly_treemap,
            "Sunburst Chart": plotly_sunburst,
            "Funnel Chart": plotly_funnel_chart,
            "Waterfall Chart": plotly_waterfall_chart,
            "Sankey Diagram": plotly_sankey_diagram,
            "Parallel Coordinates Plot": plotly_parallel_coordinates,
            "Radar Chart": plotly_radar_chart,
            "Gauge Chart": plotly_gauge_chart,
        }
    }
    
    # Display selected visualization
    st.header(f"📊 {library} - {graph_type}")
    st.markdown(f"*Graph created from **preprocessed** heart disease dataset*")
    st.markdown("---")
    
    # Execute the selected function
    try:
        function_map[library][graph_type]()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    # Display dataset information
    with st.expander("ℹ️ Dataset Information (Preprocessed Data)"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Columns", len(data.columns))
        
        st.subheader("📋 Preprocessed Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        st.subheader("📊 Statistical Summary (Preprocessed Data)")
        st.dataframe(data.describe(), use_container_width=True)

else:
    st.error("Unable to load data. Please ensure 'heart.csv' is in the correct directory.")
