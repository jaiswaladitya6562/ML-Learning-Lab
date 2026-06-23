import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==================================================
# TITLE
# ==================================================

st.title("Decision Tree")

st.caption(
    "Learn how Decision Trees split data and make predictions through a sequence of decisions."
)

# ==================================================
# THEORY
# ==================================================

st.header("What is a Decision Tree?")

st.write("""
A Decision Tree is a supervised machine learning algorithm used for
classification and regression tasks.

It makes predictions by repeatedly splitting data into smaller groups
based on feature values until a final decision is reached.
""")

st.header("How Does It Work?")

st.write("""
A Decision Tree asks a sequence of questions.

Example:

Income > 50000?

→ Yes

Credit Score > 700?

→ Yes

Loan Approved

Each split helps reduce uncertainty and move closer to a prediction.
""")

st.header("Entropy")

st.latex(
    r"Entropy=-\sum p_i\log_2(p_i)"
)

st.write("""
Entropy measures uncertainty in a dataset.

Low Entropy:
- Mostly one class
- Easy decision

High Entropy:
- Mixed classes
- Difficult decision
""")

st.header("Information Gain")

st.write("""
Information Gain measures how much uncertainty is reduced after a split.

The Decision Tree chooses the split that provides the highest
Information Gain.
""")

st.header("Real World Applications")

st.markdown("""
- Loan Approval Systems
- Medical Diagnosis
- Fraud Detection
- Customer Churn Prediction
- Risk Assessment
""")

st.divider()

# ==================================================
# DATASET
# ==================================================

st.header("Dataset")

df = pd.read_csv(
    "datasets/Loan_Approval.csv"
)

st.dataframe(
    df,
    use_container_width=True
)

X = df[
    ["Age", "Income", "CreditScore"]
]

y = df["Approved"]

st.divider()

# ==================================================
# SESSION STATE
# ==================================================

if "decision_tree_trained" not in st.session_state:
    st.session_state.decision_tree_trained = False

if "decision_tree_model" not in st.session_state:
    st.session_state.decision_tree_model = None

# ==================================================
# MODEL TRAINING
# ==================================================

st.header("Model Training")

max_depth = st.slider(
    "Maximum Tree Depth",
    min_value=1,
    max_value=10,
    value=4
)

if st.button(
    "Train Model",
    type="primary"
):

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X, y)

    st.session_state.decision_tree_model = model
    st.session_state.decision_tree_trained = True

# ==================================================
# RESULTS
# ==================================================

if st.session_state.decision_tree_trained:

    model = st.session_state.decision_tree_model

    predictions = model.predict(X)

    st.divider()

    st.header("Results")

    st.success(
        "Model trained successfully!"
    )

    # ==============================================
    # METRICS
    # ==============================================

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions
    )

    recall = recall_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    st.subheader(
        "Performance Metrics"
    )

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
    # CONFUSION MATRIX
    # ==============================================

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    TN, FP, FN, TP = cm.ravel()

    cm_df = pd.DataFrame(
        {
            "Predicted Rejected": [
                TN,
                FN
            ],
            "Predicted Approved": [
                FP,
                TP
            ]
        },
        index=[
            "Actual Rejected",
            "Actual Approved"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # ==============================================
    # TREE VISUALIZATION
    # ==============================================

    st.subheader(
        "Decision Tree Visualization"
    )

    fig, ax = plt.subplots(
        figsize=(18, 10)
    )

    plot_tree(
        model,
        feature_names=X.columns,
        class_names=[
            "Rejected",
            "Approved"
        ],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )

    st.pyplot(fig)

    # ==============================================
    # PLAYGROUND
    # ==============================================

    st.divider()

    st.header(
        "Interactive Loan Approval Predictor"
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=35
    )

    income = st.number_input(
        "Income",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=1000
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

    user_input = [
        [
            age,
            income,
            credit_score
        ]
    ]

    prediction = model.predict(
        user_input
    )[0]

    prediction_text = (
        "APPROVED"
        if prediction == 1
        else "REJECTED"
    )

    st.metric(
        "Prediction",
        prediction_text
    )

    # ==============================================
    # DECISION PATH
    # ==============================================

    st.subheader(
        "Decision Path Analysis"
    )

    node_indicator = model.decision_path(
        user_input
    )

    leave_id = model.apply(
        user_input
    )

    feature_names = X.columns

    for node_id in node_indicator.indices:

        if leave_id[0] == node_id:
            continue

        feature_index = model.tree_.feature[
            node_id
        ]

        threshold = model.tree_.threshold[
            node_id
        ]

        if feature_index == -2:
            continue

        feature_name = feature_names[
            feature_index
        ]

        user_value = user_input[0][
            feature_index
        ]

        if user_value <= threshold:

            st.write(
                f"✔ {feature_name} ({user_value}) ≤ {threshold:.2f}"
            )

        else:

            st.write(
                f"✔ {feature_name} ({user_value}) > {threshold:.2f}"
            )

    st.success(
        f"Final Prediction: {prediction_text}"
    )

    # ==============================================
    # INTERPRETATION
    # ==============================================

    st.subheader(
        "How the Tree Made Its Decision"
    )

    st.write("""
The Decision Tree repeatedly split the data using
feature thresholds that reduced uncertainty.

The decision path above shows exactly how your input
traveled through the tree before reaching the final
prediction.

Unlike many machine learning models, Decision Trees
are highly interpretable because every decision can
be traced and explained.
""")