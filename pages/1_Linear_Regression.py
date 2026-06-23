import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==================================================
# TITLE
# ==================================================

st.title("Linear Regression")

st.caption(
    "Learn the concept, train a model, visualize results, and make predictions."
)

# ==================================================
# THEORY
# ==================================================

st.header("What is Linear Regression?")

st.write("""
Linear Regression is a supervised machine learning algorithm used
to predict continuous numerical values by learning relationships
between input features and an output variable.
""")

st.header("How Does It Work?")

st.write("""
The model learns a relationship between the independent variables
(features) and the dependent variable (target).

During training, it finds the line that best fits the data by
minimizing prediction error.
""")

st.header("Mathematical Formula")

st.latex(r"y = mx + b")

st.write("""
- **y** → Predicted value
- **m** → Slope of the line
- **b** → Intercept
""")

st.header("Real World Applications")

st.markdown("""
- House Price Prediction
- Sales Forecasting
- Revenue Estimation
- Demand Prediction
- Trend Analysis
""")

st.divider()

# ==================================================
# DATASET
# ==================================================

st.header("Dataset - For House Price Prediction")

df = pd.read_csv("datasets/Student_Scores.csv")

st.dataframe(df, use_container_width=True)

# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.divider()

# ==================================================
# SESSION STATE
# ==================================================

if "trained" not in st.session_state:
    st.session_state.trained = False

if "model" not in st.session_state:
    st.session_state.model = None

# ==================================================
# TRAIN MODEL
# ==================================================

st.header("Model Training")

if st.button("Train Model", type="primary"):

    model = LinearRegression()

    model.fit(X, y)

    st.session_state.model = model
    st.session_state.trained = True

# ==================================================
# RESULTS
# ==================================================

if st.session_state.trained:

    model = st.session_state.model

    predictions = model.predict(X)

    st.divider()

    st.header("Results")

    st.success("Model trained successfully!")

    # ==============================================
    # METRICS
    # ==============================================

    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    with col2:
        st.metric(
            "Mean Squared Error",
            f"{mse:.3f}"
        )

    # ==============================================
    # VISUALIZATION
    # ==============================================

    st.subheader("Area vs Price")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df.iloc[:, 0],
        y
    )

    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(y.name)
    ax.set_title(f"{X.columns[0]} vs {y.name}")

    st.pyplot(fig)

    # ==============================================
    # MODEL INSIGHTS
    # ==============================================

    st.subheader("Model Insights")

    st.write(
        f"""
        The model uses **{len(X.columns)} feature(s)** to estimate
        **{y.name}**.

        In this dataset, larger houses and houses with more rooms
        generally have higher prices.
        """
    )

    # ==============================================
    # PLAYGROUND
    # ==============================================

    st.divider()

    st.header("Interactive House Price Predictor")

    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        value=1500,
        step=100
    )

    rooms = st.number_input(
        "Number of Rooms",
        min_value=1,
        value=3,
        step=1
    )

    predicted_price = model.predict([[area, rooms]])[0]

    st.success(
        f"Estimated House Price: ₹{predicted_price:.2f} Lakhs"
    )