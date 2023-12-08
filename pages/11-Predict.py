import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the trained logistic regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.coef_ = np.array([[-0.45735262, 0.18649083, -0.03763093, -0.23950828, 0.05404476, -0.08057588,
                                0.08625252, 0.17357327, -0.11847064, -0.04612795, 1.25295737]])

logreg_model.intercept_ = np.array([-0.10373654])
logreg_model.classes_ = np.array([0, 1])

# Streamlit app display
st.title("Loan Status Prediction Using Logistic Regression")

# User input for prediction
st.sidebar.header("User Input:")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.slider("Applicant Income", 0, 80000, 60000)
coapplicant_income = st.sidebar.slider("Coapplicant Income", 0, 50000, 40000)
loan_amount = st.sidebar.slider("Loan Amount", 0, 800, 200)
loan_amount_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", ["0", "1"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert user input to numeric values
to_numeric = {'Male': 1, 'Female': 2,
              'Yes': 1, 'No': 2,
              'Graduate': 1, 'Not Graduate': 2,
              'Urban': 3, 'Semiurban': 2, 'Rural': 1,
              '1': 1, '2': 2, '3+': 3,
              '0': 0, '1': 1}
user_input = pd.DataFrame({
    'Gender': [to_numeric[gender]],
    'Married': [to_numeric[married]],
    'Dependents': [to_numeric[dependents]],
    'Education': [to_numeric[education]],
    'Self_Employed': [to_numeric[self_employed]],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [to_numeric[credit_history]],
    'Property_Area': [to_numeric[property_area]]
})

# Make prediction
user_input_scaled = StandardScaler().fit_transform(user_input)
prediction = logreg_model.predict(user_input_scaled)

# Display prediction result
st.header("Prediction Result:")
if prediction[0] == 1:
    st.success("Loan Approved!")
else:
    st.error("Loan Not Approved.")
