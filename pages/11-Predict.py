import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
@st.cache  # Cache the dataset to avoid reloading on each Streamlit rerun
def load_data():
    data = pd.read_csv('train.csv')
    return data

# Data preprocessing functions
def to_numeric(df):
    to_numeric = {'Male': 1, 'Female': 2,
                  'Yes': 1, 'No': 2,
                  'Graduate': 1, 'Not Graduate': 2,
                  'Urban': 3, 'Semiurban': 2, 'Rural': 1,
                  'Y': 1, 'N': 0,
                  '3+': 3}
    return df.applymap(lambda label: to_numeric.get(label, label))

def fill_data_mode(df):
    null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount',
                 'Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
    for col in null_cols:
        df[col].fillna(df[col].dropna().mode().values[0], inplace=True)
    return df

# Streamlit app display
st.title("Loan Status Prediction!")

# Load and preprocess data
data = load_data()
df = data.drop('Loan_ID', axis=1)
df_num = to_numeric(df)
df_num_mode = fill_data_mode(df_num)

# Split the data into features (X) and target variable (y)
y = df_num_mode['Loan_Status']
X = df_num_mode.drop('Loan_Status', axis=1)

# Split the data into training and testing sets
test_fraction = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a logistic regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train_scaled, y_train)

# User input for prediction
st.sidebar.header("User Input:")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.slider("Applicant Income", 0, 80000, 0)
coapplicant_income = st.sidebar.slider("Coapplicant Income", 0, 50000, 0)
loan_amount = st.sidebar.slider("Loan Amount", 0, 800, 200)
loan_amount_term = st.sidebar.slider("Loan Amount Term", 0, 500, 200)
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
user_input_scaled = scaler.transform(user_input)
prediction = logreg_model.predict(user_input_scaled)

# Display prediction result
st.header("Prediction Result:")
if prediction[0] == 1:
    st.success("Loan Approved!")
else:
    st.error("Loan Not Approved.")
