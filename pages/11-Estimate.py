import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.naive_bayes import GaussianNB  # Import Naive Bayes Classifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


# Load the dataset
def load_data():
    data = pd.read_csv('train.csv')
    return data

def to_numeric(df):
    to_numeric = {'Male': 1, 'Female': 2,
                  'Yes': 1, 'No': 2,
                  'Graduate': 1, 'Not Graduate': 2,
                  'Urban': 3, 'Semiurban': 2, 'Rural': 1,
                  'Y': 1, 'N': 0,
                  '3+': 3}

    data = df.applymap(lambda label: to_numeric.get(label) if label in to_numeric else label)
    return data

def fill_data_mode(df):
    null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount', 'Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
    for col in null_cols:
        df[col] = df[col].fillna(df[col].dropna().mode().values[0])
    return df

# Streamlit app display
st.title("Loan Status Prediction with Different Models")

# Load and preprocess data
data = load_data()
df = data.drop('Loan_ID', axis=1)
df_num = to_numeric(df)
df_num_mode = fill_data_mode(df_num)

# Define models with best hyperparameters
models = {
    'Logistic Regression': LogisticRegression(random_state=start_state),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=start_state),
    'k-Nearest Neighbor': KNeighborsClassifier(n_neighbors=9),
    'Decision Tree': DecisionTreeClassifier(max_depth=1, random_state=start_state),
    'Support Vector Classifier': SVC(random_state=start_state),  # You can choose different kernels
    'Naive Bayes': GaussianNB()
}

# Select model from dropdown
selected_model = st.selectbox("Select Model", list(models.keys()) if models else [])


# Set parameters from the dataset
default_parameters = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 0,
    'LoanAmount': 120,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 'Urban',
}

# Collect parameter values from the user
parameters = {}
for param, default_value in default_parameters.items():
    if param in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        parameters[param] = st.selectbox(param, df_num_mode[param].unique(), index=df_num_mode[param].unique().tolist().index(default_value))
    else:
        parameters[param] = st.number_input(param, value=default_value)

# Split the data into features (X) and target variable (y)
y = df_num_mode['Loan_Status']
X = df_num_mode.drop('Loan_Status', axis=1)

# Split the data into training and testing sets
start_state = 42
test_fraction = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose the selected model
model = models[selected_model]

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the input parameters
input_data = pd.DataFrame(parameters, index=[0])
input_data = to_numeric(fill_data_mode(input_data))
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)[0]

# Display the prediction
st.subheader("Loan Status Prediction:")
st.write(f"The model predicts the loan status as: {prediction}")

# Display additional information if desired
if st.checkbox("Show Detailed Information"):
    st.subheader("Detailed Information:")
    st.write("Accuracy on Test Set:", accuracy_score(y_test, model.predict(X_test_scaled)))
    st.write("Classification Report:")
    st.text(classification_report(y_test, model.predict(X_test_scaled)))
    st.write("Confusion Matrix:")
    conf_mat = confusion_matrix(y_test, model.predict(X_test_scaled))
    ConfusionMatrixDisplay(conf_mat, display_labels=['Not Approved', 'Approved']).plot()
    st.pyplot()

# Optional: Display the input parameters
if st.checkbox("Show Input Parameters"):
    st.subheader("Input Parameters:")
    st.write(input_data)
