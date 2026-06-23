import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# ==================================================
# TITLE
# ==================================================

st.title("K-Means Clustering")

st.caption(
    "Learn how K-Means discovers patterns in unlabeled data."
)

# ==================================================
# THEORY
# ==================================================

st.header("What is Unsupervised Learning?")

st.write("""
Unlike supervised learning, unsupervised learning does not use
labeled data.

In supervised learning, the algorithm is provided with both
input features and the correct answers.

In unsupervised learning, the algorithm receives only the input
features and must discover patterns, structures, or groups on
its own.

Examples include:

• Customer Segmentation

• Market Basket Analysis

• Recommendation Systems

• Anomaly Detection

K-Means is one of the most popular unsupervised learning algorithms.
""")

st.header("What is K-Means?")

st.write("""
K-Means is a clustering algorithm that groups similar data points
together.

The algorithm attempts to divide a dataset into K clusters such
that points within a cluster are similar while points belonging
to different clusters are dissimilar.

Unlike previous algorithms such as Logistic Regression,
Decision Trees, or KNN, K-Means does not predict labels.
Instead, it discovers groups automatically.
""")

st.header("How Does K-Means Work?")

st.write("""
1. Choose the number of clusters (K)

2. Randomly initialize cluster centroids

3. Assign every data point to the nearest centroid

4. Recalculate centroid positions

5. Repeat until centroids stop moving
""")

st.header("Why Does Choosing K Matter?")

st.write("""
A very small K may merge different groups together.

A very large K may create unnecessary clusters.

Selecting the correct K is one of the most important parts
of K-Means clustering.
""")

st.header("Real World Applications")

st.markdown("""
- Customer Segmentation
- Product Recommendations
- Market Analysis
- Image Compression
- Fraud Detection
- User Behaviour Analysis
""")

st.divider()

# ==================================================
# DATASET
# ==================================================

st.header("Dataset")

df = pd.read_csv(
    "datasets/Customer_Segmentation.csv"
)

st.dataframe(
    df,
    use_container_width=True
)

# ==================================================
# FEATURES
# ==================================================

X = df[
    [
        "AnnualIncome",
        "SpendingScore"
    ]
]

st.divider()

# ==================================================
# K SELECTION
# ==================================================

st.header("Cluster Configuration")

k = st.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=8,
    value=4
)

# ==================================================
# MODEL
# ==================================================

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X)

df["Cluster"] = clusters

# ==================================================
# CLUSTER VISUALIZATION
# ==================================================

st.header("Cluster Visualization")

fig, ax = plt.subplots(
    figsize=(8, 6)
)

scatter = ax.scatter(
    df["AnnualIncome"],
    df["SpendingScore"],
    c=df["Cluster"],
    alpha=0.8
)

ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=300,
    c="black",
    label="Centroids"
)

ax.set_xlabel(
    "Annual Income"
)

ax.set_ylabel(
    "Spending Score"
)

ax.set_title(
    f"K-Means Clustering (K = {k})"
)

ax.legend()

st.pyplot(fig)

# ==================================================
# CLUSTER CENTERS
# ==================================================

st.header("Cluster Centers")

centroids = pd.DataFrame(
    model.cluster_centers_,
    columns=[
        "AnnualIncome",
        "SpendingScore"
    ]
)

centroids.index = [
    f"Cluster {i}"
    for i in range(k)
]

st.dataframe(
    centroids,
    use_container_width=True
)

# ==================================================
# CLUSTER DISTRIBUTION
# ==================================================

st.header("Cluster Distribution")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(
    cluster_counts
)

# ==================================================
# ELBOW METHOD
# ==================================================

st.header("Elbow Method")

st.write("""
The Elbow Method helps determine a suitable value of K.

As K increases, clustering improves and inertia decreases.

The optimal K is often located at the "elbow" point where
additional clusters provide diminishing returns.
""")

inertias = []

for i in range(1, 11):

    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(X)

    inertias.append(
        km.inertia_
    )

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    range(1, 11),
    inertias,
    marker="o"
)

ax.set_xlabel(
    "Number of Clusters (K)"
)

ax.set_ylabel(
    "Inertia"
)

ax.set_title(
    "Elbow Method"
)

st.pyplot(fig)

# ==================================================
# PLAYGROUND
# ==================================================

st.divider()

st.header(
    "Interactive Customer Analysis"
)

income = st.number_input(
    "Annual Income",
    min_value=10000,
    max_value=200000,
    value=60000,
    step=1000
)

spending = st.number_input(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

customer = np.array(
    [[income, spending]]
)

cluster = model.predict(
    customer
)[0]

st.metric(
    "Assigned Cluster",
    cluster
)

# ==================================================
# USER VISUALIZATION
# ==================================================

st.subheader(
    "Customer Position in Cluster Space"
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.scatter(
    df["AnnualIncome"],
    df["SpendingScore"],
    c=df["Cluster"],
    alpha=0.7
)

ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=300,
    c="black",
    label="Centroids"
)

ax.scatter(
    income,
    spending,
    marker="*",
    s=400,
    c="red",
    label="Your Customer"
)

ax.set_xlabel(
    "Annual Income"
)

ax.set_ylabel(
    "Spending Score"
)

ax.set_title(
    "Customer Cluster Assignment"
)

ax.legend()

st.pyplot(fig)


# ==================================================
# CLUSTER PROFILE
# ==================================================

st.subheader(
    "Cluster Profile"
)

cluster_df = df[
    df["Cluster"] == cluster
]

cluster_size = (
    df["Cluster"] == cluster
).sum()

avg_income = cluster_df["AnnualIncome"].mean()
avg_spending = cluster_df["SpendingScore"].mean()

min_income = cluster_df["AnnualIncome"].min()
max_income = cluster_df["AnnualIncome"].max()

min_spending = cluster_df["SpendingScore"].min()
max_spending = cluster_df["SpendingScore"].max()

cluster_share = (
    cluster_size / len(df)
) * 100

st.write(
    f"""
**Average Annual Income:** {avg_income:,.0f}

**Average Spending Score:** {avg_spending:,.1f}

**Annual Income Range:** {min_income:,.0f} – {max_income:,.0f}

**Spending Score Range:** {min_spending:,.0f} – {max_spending:,.0f}

**Share of Total Customers:** {cluster_share:.1f}% ({cluster_size} of {len(df)})

Comparing your customer's income (**{income:,.0f}**) and spending
score (**{spending}**) against these cluster averages shows how
typical or unusual they are within Cluster {cluster}.
"""
)

# ==================================================
# INTERPRETATION
# ==================================================

st.subheader(
    "Cluster Interpretation"
)

st.write(
    f"""
The customer has been assigned to **Cluster {cluster}**.

This cluster currently contains **{cluster_size} customers**
with similar income levels and spending behaviour.

Try changing:

• Annual Income

• Spending Score

• Number of Clusters (K)

and observe how the customer's cluster assignment changes.

Unlike supervised learning algorithms, K-Means is not predicting
an answer. Instead, it is discovering groups that naturally exist
within the dataset.
"""
)