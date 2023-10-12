!pip install streamlit

import streamlit as st
import pickle  # To load the trained model
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the Streamlit app
st.title("Machine Learning Model Deployment")

# Create input elements (e.g., text input, sliders)
feature1 = st.slider("Feature 1", 0.0, 10.0)
feature2 = st.slider("Feature 2", 0.0, 10.0)

# Create a button to make predictions
if st.button("Predict"):
    # Make predictions using the loaded model
    input_data = np.array([feature1, feature2]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
