# Imports
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import streamlit as st

# Write title and description
st.title("Task 2 - Benchmarking ML Algorithms")

# Store initial variable values in session state
if "visibility" not in st.session_state:
    st.session_state.selection = "Decision Trees"

# Request user input
st.radio(
    "Select machine learning algorithm",
    ["Decision Trees", "Logistic Regression", "Support Vector Machine"],
    key="selection"
)

# Fetch dataset
dataset_car_evaluation = fetch_ucirepo(id=19)

# Initiate data variables
X = dataset_car_evaluation.data.features
y = dataset_car_evaluation.data.targets

# Print data analysis
st.write(f"For this example, we are using the [Car Evaluation dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)")
st.write()
st.write(f"The dataset has {str(len(X.columns))} features")
st.write(f"The dataset has {str(len(X))} instances")
st.write()
st.write("The dataset has the following first 5 rows:")
st.write(X.head())
st.write()
st.write("The dataset has the following specifications:")
st.write(X.describe())
st.write()
st.write("The dataset has the following empty values:")
st.write(X.isnull().sum())

# Split data into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Encode features datasets using pandas
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Create model depending on the chosen algorithm
if st.session_state.selection == "Decision Trees":
    model = DecisionTreeClassifier()
elif st.session_state.selection == "Logistic Regression":
    model = LogisticRegression()
else:
    model = SVC()

# Fit train data into model
model.fit(X_train_encoded, np.ravel(y_train))

# Create predictions for model based on test data
predictions = model.predict(X_test_encoded)

# Determine accuracy for predictions based on test data
accuracy = accuracy_score(y_test, predictions)

# Write accuracy to streamlit application
st.write(f"The accuracy for this model is {np.round(accuracy * 100, 2)}")