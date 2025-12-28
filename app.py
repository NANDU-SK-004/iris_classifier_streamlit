import streamlit as st
import pickle
from sklearn.datasets import load_iris

# Load Iris dataset class names
iris = load_iris()
target_names = iris.target_names

# Load trained model from .pkl file
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title and description
st.title("🌼 Iris Flower Classifier")
st.write("Enter flower measurements to predict its species using a Decision Tree model.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    st.success(f"🌸 Predicted Species: {target_names[prediction].capitalize()}")
