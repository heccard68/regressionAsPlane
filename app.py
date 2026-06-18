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
        """, unsafe_allow_html=True)
        
    else:
        st.success("""
        **Current View: Multiple Linear Regression ($Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2$)**
        
        Now, we pull the data out into a third dimension using $X_2$ (Bedrooms). Look at how the points spread out into depth!
        
        Because the data exists in 3D space, a 1D line can no longer capture the trend. The model must construct a **2D Plane** to minimize the residuals in this 3D room.
        """)
        
        # Calculate 3D Regression Plane using Ordinary Least Squares formula
        X_mat = np.vstack([np.ones(n_points), X1, X2]).T
        beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
        
        # Pre-format the text equation to avoid f-string HTML formatting confusion
        eq_text_3d = f"Price = {beta[0]:.2f} + {beta[1]:.4f} × (SqFt) + {beta[2]:.2f} × (Bedrooms)"
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <p style="margin: 0; font-size: 14px; color: #555; font-weight: bold;">Regression Equation</p>
            <p style="margin: 0; font-size: 16px; font-weight: bold; color: #28a745; word-wrap: break-word;">
                {eq_text_3d}
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if dimension == "2D View (Ignore $X_2$)":
        # 2D Scatter plot with Trendline (Requires statsmodels in requirements.txt)
        fig_2d = px.scatter(
            df, 
            x="SquareFootage ($X_1$)", 
            y="Price ($Y$)", 
            trendline="ols",
            trendline_color_override="red",
            title="2D Space: Line of Best Fit"
        )
        fig_2d.update_layout(height=600)
        st.plotly_chart(fig_2d, use_container_width=True)
        
    else:
        # 3D Scatter plot with Surface Plane
        # Create a grid for the plane
        x1_range = np.linspace(X1.min(), X1.max(), 20)
        x2_range = np.linspace(X2.min(), X2.max(), 20)
        X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
        
        # Calculate predicted Y values for the grid surface
        Y_grid = beta[0] + beta[1] * X1_grid + beta[2] * X2_grid
        
        # Build the 3D Plotly Figure
        fig_3d = go.Figure()
        
        # Add the raw data points
        fig_3d.add_trace(go.Scatter3d(
            x=df["SquareFootage ($X_1$)"],
            y=df["Bedrooms ($X_2$)"],
            z=df["Price ($Y$)"],
            mode='markers',
            marker=dict(size=4, color=df["Price ($Y$)"], colorscale='Viridis', opacity=0.8),
            name="Data Points"
        ))
        
        # Add the regression plane
        fig_3d.add_trace(go.Surface(
            x=x1_range,
            y=x2_range,
            z=Y_grid,
            colorscale='Reds',
            opacity=0.6,
            showscale=False,
            name="Regression Plane"
        ))
        
        fig_3d.update_layout(
            title="3D Space: Plane of Best Fit (Click & Drag to Rotate!)",
            scene=dict(
                xaxis_title='Square Footage (X1)',
                yaxis_title='Bedrooms (X2)',
                zaxis_title='Price (Y)'
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

st.info("""
💡 **Teacher's Tip for Class:** Tell your students to toggle back and forth between the 2D and 3D views. 
In the **3D View**, they can **click and drag the graph to rotate it**. 
If they rotate the 3D graph so they are looking perfectly down the side edge of the 'Bedrooms' axis, they will see the 3D plane flatten back down into the exact 2D regression line they saw in the first view!
""")
