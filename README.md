# From Regression Line to Regression Plane

This Streamlit app is designed for a public policy research methods class.

It provides a live visualization of the transition from:

- **Bivariate regression**: one explanatory variable, one fitted line
- **Multivariate regression**: two explanatory variables, one fitted plane

The goal is to help students understand the intuition behind the phrase:

> Holding other variables constant

## Files

- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — optional Streamlit theme/config

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Go to https://share.streamlit.io/
4. Connect your GitHub account.
5. Select the repository.
6. Set the main file path to:

```text
app.py
```

7. Deploy.

## Teaching notes

Use the app in this order:

1. Start with the bivariate line.
2. Ask students what is missing.
3. Move to the multivariate plane.
4. Explain that "holding X2 constant" means moving along the plane in the X1 direction while keeping X2 fixed.
5. Connect this to omitted variable bias.

## Important note

The data are simulated for teaching purposes. They are not empirical estimates of income, education, or experience.
