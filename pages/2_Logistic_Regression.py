import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
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

st.title("Logistic Regression")

st.caption(
    "Learn how classification models predict probabilities and make decisions."
)

# ==================================================
# THEORY
# ==================================================

st.header("What is Logistic Regression?")

st.write("""
Logistic Regression is a supervised machine learning algorithm used
for classification tasks. Unlike Linear Regression, which predicts
continuous values, Logistic Regression predicts the probability that
an observation belongs to a particular class.
""")

st.header("Why Not Linear Regression?")

st.write("""
If Linear Regression is used for classification, the output can be
any number such as -1.5 or 2.3, which does not make sense when
predicting classes like Spam or Not Spam.

Logistic Regression solves this problem by converting outputs into
probabilities between 0 and 1.
""")

st.header("Sigmoid Function")

st.latex(r"P(Y=1)=\frac{1}{1+e^{-z}}")

st.write("""
The Sigmoid Function transforms any input into a value between 0 and 1.

The output represents the probability of belonging to the positive class.
""")

st.header("Spam Detection Example")

st.write("""
One of the most common applications of Logistic Regression is email
spam detection.

The model examines features such as:

• Number of suspicious links

• Number of spam-related words

• Presence of attachments

• Excessive capitalization

Based on these features, it predicts the probability that an email is spam.
""")

st.header("Decision Boundary")

st.write("""
The model predicts a probability rather than directly predicting Spam
or Not Spam.

A threshold is then used to make the final decision.
""")

st.write("""
Default Threshold:

Probability ≥ 0.50 → Spam

Probability < 0.50 → Not Spam
""")

st.header("Why Use a Higher Threshold?")

st.write("""
Suppose an email receives a spam probability of 0.60.

Using a threshold of 0.50 would classify it as Spam.

However, falsely marking an important email as spam can be costly.
For example:

• Job offers

• Bank notifications

• College admission emails

To reduce such false positives, many systems increase the threshold
to 0.70 or higher.

This means only emails with strong evidence of being spam are blocked.
""")

st.divider()

# ==================================================
# DATASET
# ==================================================

st.header("Dataset")

df = pd.read_csv("datasets/Email_spam.csv")

st.dataframe(df, use_container_width=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.divider()

# ==================================================
# SESSION STATE
# ==================================================

if "logistic_trained" not in st.session_state:
    st.session_state.logistic_trained = False

if "logistic_model" not in st.session_state:
    st.session_state.logistic_model = None

# ==================================================
# TRAIN MODEL
# ==================================================

st.header("Model Training")

if st.button("Train Model", type="primary"):

    model = LogisticRegression()

    model.fit(X, y)

    st.session_state.logistic_model = model
    st.session_state.logistic_trained = True

# ==================================================
# RESULTS
# ==================================================

if st.session_state.logistic_trained:

    model = st.session_state.logistic_model

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
    # CONFUSION MATRIX
    # ==============================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Not Spam", "Actual Spam"],
        columns=["Predicted Not Spam", "Predicted Spam"]
    )

    st.dataframe(cm_df)

    # ==============================================
    # DATA VISUALIZATION
    # ==============================================

    st.subheader("Dataset Visualization")

    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        df.iloc[:, 0],
        df.iloc[:, 1],
        c=df.iloc[:, -1]
    )

    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_title("Spam vs Not Spam")

    st.pyplot(fig)

    # ==============================================
    # PLAYGROUND
    # ==============================================

    st.divider()

    st.header("Interactive Spam Classifier")

    links = st.number_input(
        "Number of Suspicious Links",
        min_value=0,
        value=3,
        step=1
    )

    spam_words = st.number_input(
        "Number of Spam Keywords",
        min_value=0,
        value=5,
        step=1
    )

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.70,
        step=0.05
    )

    # ==============================================
    # SINGLE EMAIL PREDICTION
    # ==============================================

    probability = model.predict_proba(
        [[links, spam_words]]
    )[0][1]

    prediction = (
        "SPAM"
        if probability >= threshold
        else "NOT SPAM"
    )

    st.subheader("Current Email Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Spam Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Prediction",
            prediction
        )

    # ==============================================
    # THRESHOLD-BASED CONFUSION MATRIX
    # ==============================================

    st.subheader("Threshold-Based Confusion Matrix")

    all_probabilities = model.predict_proba(X)[:, 1]

    custom_predictions = (
        all_probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y,
        custom_predictions
    )

    TN, FP, FN, TP = cm.ravel()

    cm_df = pd.DataFrame(
        {
            "Predicted Not Spam": [TN, FN],
            "Predicted Spam": [FP, TP]
        },
        index=[
            "Actual Not Spam",
            "Actual Spam"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # ==============================================
    # CONFUSION MATRIX BREAKDOWN
    # ==============================================

    st.subheader("Classification Outcomes")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "True Positives",
            TP
        )

        st.metric(
            "False Positives",
            FP
        )

    with col2:
        st.metric(
            "True Negatives",
            TN
        )

        st.metric(
            "False Negatives",
            FN
        )

    # ==============================================
    # THRESHOLD INTERPRETATION
    # ==============================================

    st.subheader("Threshold Interpretation")

    if threshold < 0.5:

        st.warning("""
    ### Aggressive Spam Filtering

    The selected threshold is relatively low.

    **Effect:**
    - More emails are classified as spam
    - Higher spam detection rate
    - Increased False Positives
    - Legitimate emails may be blocked
    """)

    elif threshold < 0.7:

        st.info("""
    ### Balanced Filtering

    The selected threshold provides a balance between
    spam detection and avoiding incorrect classifications.

    **Effect:**
    - Moderate False Positives
    - Moderate False Negatives
    - Common default choice
    """)

    else:

        st.success("""
    ### Conservative Spam Filtering

    The selected threshold is relatively high.

    **Effect:**
    - Fewer False Positives
    - Important emails are less likely to be blocked
    - More spam emails may reach the inbox
    """)