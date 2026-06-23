import streamlit as st

st.set_page_config(page_title="ML Learning Lab")

st.title("ML Learning Lab")
st.write("""
Machine Learning is a branch of Artificial Intelligence that allows
computers to learn patterns from data and make predictions or
decisions without being explicitly programmed for every rule.

Instead of following fixed instructions, a machine learning model
improves its performance by learning from examples.
         

**Choose a model from the sidebar.**

         
**Linear Regression**

Predicts a continuous numeric value by fitting a straight line
through the data. It assumes a linear relationship between the
input features and the output.

**Logistic Regression**

Despite the name, it is used for classification, not regression.
It predicts the probability that a data point belongs to a
particular class using an S-shaped (sigmoid) curve.

**K-Nearest Neighbours (KNN)**

Classifies a new data point based on the majority class among its
"K" closest neighbours in the dataset. It does not build a model
during training; instead, it makes decisions at prediction time.

**Decision Trees**

Splits data into branches using a series of yes/no questions based
on feature values, forming a tree-like structure. The result is
easy to visualize and interpret, much like a flowchart.

**K-Means Clustering**

An unsupervised algorithm that groups unlabeled data into clusters
based on similarity. Unlike the other models here, it does not
predict labels — it discovers structure in the data on its own.
""")
