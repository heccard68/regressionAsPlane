import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📉 Dimension Shift: Line to Regression Plane")
st.markdown("""
### Visualizing the Transition from 2D to 3D Linear Regression
In introductory statistics, students learn 2D regression: predicting an outcome ($Y$) using one variable ($X_1$). 
But when we introduce a second predictor ($X_2$), we shift dimensions. The line of best fit becomes a **plane of best fit**.
""")

# 1. Generate Synthetic Data (Predicting House Price based on Size and Bedrooms)
np.random.seed(42)
n_points = 150
X1 = np.random.uniform(1000, 3500, n_points)  # e.g., Square Footage
X2 = np.random.uniform(1, 5, n_points)       # e.g., Number of Bedrooms
# True relationship + some noise
Y = 50 + 0.12 * X1 + 25 * X2 + np.random.normal(0, 30, n_points) 

df = pd.DataFrame({"SquareFootage ($X_1$)": X1, "Bedrooms ($X_2$)": X2, "Price ($Y$)": Y})

# Sidebar Controls
st.sidebar.header("Visualization Mode")
dimension = st.sidebar.radio(
    "Select Dimension View:",
    ("2D View (Ignore $X_2$)", "3D View (Include $X_2$)")
)

# Layout Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("The Concept")
    if dimension == "2D View (Ignore $X_2$)":
        st.info("""
        **Current View: Simple Linear Regression ($Y = \\beta_0 + \\beta_1 X_1$)**
        
        We are pretending the second variable ($X_2$) doesn't exist. This projects all data points onto a flat wall. 
        
        The model finds a **line** that minimizes the vertical distances (residuals) from the points to the line.
        """)
        
        # Calculate 2D Regression line
        b1, b0 = np.polyfit(X1, Y, 1)
        
        # Pre-format the text equation to avoid f-string HTML formatting confusion
        eq_text_2d = f"Price = {b0:.2f} + {b1:.4f} × (SqFt)"
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <p style="margin: 0; font-size: 14px; color: #555; font-weight: bold;">Regression Equation</p>
            <p style="margin: 0; font-size: 18px; font-weight: bold; color: #ff4b4b; word-wrap: break-word;">
                {eq_text_2d}
            </p>
        </div>
        """, unsafe_html=True)
        
    else:
        st.success("""
        **Current View: Multiple Linear Regression ($Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2$)**
        
        Now, we pull the data out into a third dimension using $X_2$ (Bedrooms). Look at how the points spread out into depth!
        
        Because the data exists in 3D space, a 1D line can no longer capture the trend. The model must construct a **2D Plane** to minimize the residuals in this 3D room.
        """)
        
        # Calculate 3D Regression Plane using Ordinary Least Squares formula
        X_mat = np.vstack([np.ones(n_points), X1, X2]).T
        beta = np.linalg.lstsq(X_mat,
