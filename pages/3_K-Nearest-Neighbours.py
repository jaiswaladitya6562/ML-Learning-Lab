import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================================
# TITLE
# ==================================================

st.title("K-Nearest Neighbours (KNN)")

st.caption(
    "Learn how KNN classifies new data points using nearby neighbours."
)

# ==================================================
# THEORY
# ==================================================

st.header("What is KNN?")

st.write("""
K-Nearest Neighbours (KNN) is a supervised machine learning algorithm
used for classification and regression tasks.

Instead of learning a mathematical equation, KNN stores the dataset
and classifies new data points based on the nearest neighbours.
""")

st.header("Why is KNN Called Lazy Learning?")

st.write("""
Unlike Linear Regression and Logistic Regression, KNN does not build
a model during training.

It simply stores the data and performs computations only when a
prediction is requested.
""")

st.header("Distance Calculation")

st.latex(
r"d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}"
)

st.write("""
KNN uses distance measures to identify the closest neighbours.

The nearest neighbours have the greatest influence on the prediction.
""")

st.header("Choosing K")

st.write("""
K determines how many neighbours participate in voting.

• K = 1 → Very sensitive to noise

• K = 3 → Usually balanced

• Large K → More stable but may oversimplify patterns
""")

st.header("Real World Applications")

st.markdown("""
- Product Recommendation
- Customer Segmentation
- Medical Diagnosis
- Credit Risk Analysis
- Fraud Detection
""")

st.divider()

# ==================================================
# DATASET
# ==================================================

st.header("Dataset")

df = pd.read_csv("datasets/Customer_Prediction.csv")

st.dataframe(df, use_container_width=True)

X = df[["Age", "Salary"]]
y = df["Purchased"]

st.divider()

# ==================================================
# SESSION STATE
# ==================================================

if "knn_trained" not in st.session_state:
    st.session_state.knn_trained = False

if "knn_model" not in st.session_state:
    st.session_state.knn_model = None

# ==================================================
# MODEL TRAINING
# ==================================================

st.header("Model Training")

k_value = st.slider(
    "Select K",
    min_value=1,
    max_value=9,
    value=3
)

if st.button("Train Model", type="primary"):

    model = KNeighborsClassifier(
        n_neighbors=k_value
    )

    model.fit(X, y)

    st.session_state.knn_model = model
    st.session_state.knn_trained = True

# ==================================================
# RESULTS
# ==================================================

if st.session_state.knn_trained:

    model = st.session_state.knn_model

    predictions = model.predict(X)

    st.divider()

    st.header("Results")

    st.success("Model trained successfully!")

    # ==============================================
    # METRICS
    # ==============================================

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.3f}"
        )

        st.metric(
            "Recall",
            f"{recall:.3f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision:.3f}"
        )

        st.metric(
            "F1 Score",
            f"{f1:.3f}"
        )

    # ==============================================
    # VISUALIZATION
    # ==============================================

    st.subheader("Dataset Visualization")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [
        "red" if val == 0 else "green"
        for val in y
    ]

    ax.scatter(
        df["Age"],
        df["Salary"],
        c=colors
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Salary")
    ax.set_title("Customer Purchase Dataset")

    st.pyplot(fig)

    # ==============================================
    # PLAYGROUND
    # ==============================================

    st.divider()

    st.header("Interactive Customer Predictor")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=35
    )

    salary = st.number_input(
        "Salary",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=1000
    )

    user_input = [[age, salary]]

    prediction = model.predict(user_input)[0]

    prediction_text = (
        "PURCHASE"
        if prediction == 1
        else "NOT PURCHASE"
    )

    st.metric(
        "Prediction",
        prediction_text
    )

    # ==============================================
    # NEAREST NEIGHBOURS
    # ==============================================

    st.subheader("Nearest Neighbours")

    distances, indices = model.kneighbors(user_input)

    neighbour_df = df.iloc[
        indices[0]
    ].copy()

    neighbour_df["Distance"] = distances[0]

    st.dataframe(
        neighbour_df,
        use_container_width=True
    )

    # ==============================================
    # VISUALIZATION
    # ==============================================

    st.subheader("KNN Neighbour Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [
        "red" if val == 0 else "green"
        for val in y
    ]

    ax.scatter(
        df["Age"],
        df["Salary"],
        c=colors,
        alpha=0.7
    )

    # User point
    ax.scatter(
        age,
        salary,
        marker="*",
        s=400,
        c="blue",
        label="User Input"
    )

    # Highlight nearest neighbours
    ax.scatter(
        neighbour_df["Age"],
        neighbour_df["Salary"],
        s=200,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Nearest Neighbours"
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Salary")
    ax.set_title("KNN Neighbour Visualization")

    ax.legend()

    st.pyplot(fig)

    # ==============================================
    # VOTING BREAKDOWN
    # ==============================================

    st.subheader("Voting Breakdown")

    purchase_votes = (
        neighbour_df["Purchased"] == 1
    ).sum()

    not_purchase_votes = (
        neighbour_df["Purchased"] == 0
    ).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Purchase Votes",
            purchase_votes
        )

    with col2:
        st.metric(
            "Not Purchase Votes",
            not_purchase_votes
        )

    st.progress(
        purchase_votes / k_value
    )

    st.write(
        f"Purchase Votes: {purchase_votes}/{k_value}"
    )

    st.write(
        f"""
    For K = {k_value}, the prediction is determined by majority voting
    among the {k_value} nearest neighbours.

    Try changing:

    - Age
    - Salary
    - K value

    and observe how the neighbours and prediction change.
    """
    )