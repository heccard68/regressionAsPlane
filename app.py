import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="From Regression Line to Regression Plane",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------

@st.cache_data
def generate_data(n=120, seed=42, omitted_strength=0.75, noise=6.0):
    """
    Generate teaching data where:
      y = 20 + 3*x1 + 8*x2 + noise
    and x1 and x2 are correlated.

    x1 can be read as "education"
    x2 can be read as "experience"
    y can be read as "income"

    The point is not the empirical realism of the numbers,
    but the visual logic of omitted variable bias and controls.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, n)
    x2 = omitted_strength * x1 + np.sqrt(max(0.0001, 1 - omitted_strength**2)) * rng.normal(0, 1, n)

    y = 20 + 3 * x1 + 8 * x2 + rng.normal(0, noise, n)

    df = pd.DataFrame({
        "Education_like_X1": x1,
        "Experience_like_X2": x2,
        "Income_like_Y": y
    })
    return df


def fit_models(df):
    # Bivariate: y ~ x1
    X_biv = df[["Education_like_X1"]]
    y = df["Income_like_Y"]
    biv = LinearRegression().fit(X_biv, y)

    # Multivariate: y ~ x1 + x2
    X_multi = df[["Education_like_X1", "Experience_like_X2"]]
    multi = LinearRegression().fit(X_multi, y)

    return biv, multi


def make_2d_plot(df, biv):
    x = df["Education_like_X1"].values
    y = df["Income_like_Y"].values

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = biv.intercept_ + biv.coef_[0] * x_line

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Observed data"
    ))

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Bivariate regression line"
    ))

    fig.update_layout(
        title="Bivariate regression: one X variable, one fitted line",
        xaxis_title="X1: Education-like variable",
        yaxis_title="Y: Income-like outcome",
        height=620,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    return fig


def make_3d_plot(df, multi):
    x1 = df["Education_like_X1"].values
    x2 = df["Experience_like_X2"].values
    y = df["Income_like_Y"].values

    x1_grid = np.linspace(x1.min(), x1.max(), 30)
    x2_grid = np.linspace(x2.min(), x2.max(), 30)
    X1g, X2g = np.meshgrid(x1_grid, x2_grid)

    Yg = (
        multi.intercept_
        + multi.coef_[0] * X1g
        + multi.coef_[1] * X2g
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x1,
        y=x2,
        z=y,
        mode="markers",
        name="Observed data",
        marker=dict(size=4, opacity=0.8)
    ))

    fig.add_trace(go.Surface(
        x=X1g,
        y=X2g,
        z=Yg,
        name="Multiple regression plane",
        opacity=0.55,
        showscale=False
    ))

    fig.update_layout(
        title="Multiple regression: two X variables, one fitted plane",
        scene=dict(
            xaxis_title="X1: Education-like variable",
            yaxis_title="X2: Experience-like variable",
            zaxis_title="Y: Income-like outcome"
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


# -----------------------------
# App content
# -----------------------------

st.title("From a Regression Line to a Regression Plane")

st.markdown(
    """
This interactive visualization is designed for teaching the intuition behind bivariate and multivariate regression.

**Key idea:**  
A bivariate regression fits one line through the data.  
A multivariate regression does the same basic task, but in more dimensions. With two explanatory variables, the fitted object is a **plane** rather than a line.

The data here are simulated for teaching purposes.
"""
)

with st.sidebar:
    st.header("Teaching controls")

    n = st.slider("Number of observations", 40, 300, 120, step=10)

    omitted_strength = st.slider(
        "Correlation between X1 and X2",
        0.0,
        0.95,
        0.75,
        step=0.05,
        help="Higher values make the omitted variable more strongly related to X1."
    )

    noise = st.slider(
        "Noise in the outcome",
        1.0,
        15.0,
        6.0,
        step=0.5,
        help="Higher values make the relationship harder to see."
    )

    seed = st.number_input("Random seed", value=42, step=1)

df = generate_data(n=n, seed=int(seed), omitted_strength=omitted_strength, noise=noise)
biv, multi = fit_models(df)

biv_intercept = biv.intercept_
biv_x1 = biv.coef_[0]

multi_intercept = multi.intercept_
multi_x1 = multi.coef_[0]
multi_x2 = multi.coef_[1]

st.subheader("The setup")

st.markdown(
    """
Imagine we are trying to understand income.

- **Y** = income-like outcome  
- **X1** = education-like variable  
- **X2** = experience-like variable  

The true data-generating process uses both X1 and X2.  
But the bivariate model ignores X2.
"""
)

c1, c2 = st.columns(2)

with c1:
    st.metric("Bivariate coefficient on X1", f"{biv_x1:.2f}")
    st.caption("This is the slope when we ignore X2.")

with c2:
    st.metric("Multivariate coefficient on X1", f"{multi_x1:.2f}")
    st.caption("This is the coefficient on X1 after holding X2 constant.")

st.markdown(
    f"""
**Teaching interpretation:**  
In the bivariate model, the coefficient on X1 is **{biv_x1:.2f}**.  
In the multivariate model, the coefficient on X1 is **{multi_x1:.2f}**.

The coefficient changes because X2 is correlated with X1 and also helps explain Y.  
That is the intuition behind omitted variable bias.
"""
)

tab1, tab2, tab3 = st.tabs([
    "1. Bivariate line",
    "2. Multivariate plane",
    "3. Teaching notes"
])

with tab1:
    st.plotly_chart(make_2d_plot(df, biv), use_container_width=True)
    st.markdown(
        """
**What to say in class:**  
This is the regression students already understand visually. There is one explanatory variable on the horizontal axis and one outcome variable on the vertical axis. The model fits the line that best summarizes the cloud of points.

But notice the limitation: if another variable also matters, it is hidden inside the scatter.
"""
    )

with tab2:
    st.plotly_chart(make_3d_plot(df, multi), use_container_width=True)
    st.markdown(
        """
**What to say in class:**  
Now we add a second explanatory variable. The fitted object is no longer a line. It is a plane.

The phrase "holding X2 constant" means: imagine moving along the plane in the X1 direction while staying at the same value of X2.
"""
    )

with tab3:
    st.markdown(
        """
### Suggested lecture script

1. **Start with the bivariate graph.**  
   "Here regression is doing something simple: fitting the best line through the data."

2. **Ask what might be missing.**  
   "If this were income, would education be the only thing that matters? Of course not. Experience, region, occupation, family background, and many other variables may matter."

3. **Move to the 3D graph.**  
   "When we add one more variable, the line becomes a plane. The intuition is the same, but we are now fitting the relationship across multiple dimensions."

4. **Explain holding constant.**  
   "Holding experience constant means comparing people at the same level of experience, while varying education."

5. **Connect to omitted variable bias.**  
   "If experience is correlated with education and also affects income, leaving it out can distort the education coefficient."

### Key takeaway

Bivariate regression fits a line in two dimensions.  
Multiple regression fits the equivalent of a line in higher dimensions.  
With two predictors, we can still visualize it as a plane.  
With more predictors, we cannot easily visualize it, but the logic is the same.
"""
    )

st.divider()

st.subheader("Model equations")

st.latex(r"\hat{Y} = \alpha + \beta_1 X_1")
st.write(f"Bivariate model: Ŷ = {biv_intercept:.2f} + {biv_x1:.2f}X1")

st.latex(r"\hat{Y} = \alpha + \beta_1 X_1 + \beta_2 X_2")
st.write(
    f"Multivariate model: Ŷ = {multi_intercept:.2f} "
    f"+ {multi_x1:.2f}X1 + {multi_x2:.2f}X2"
)

with st.expander("Show the simulated data"):
    st.dataframe(df)
