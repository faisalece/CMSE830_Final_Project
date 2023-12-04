import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Load the dataset
def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()

# Do not use inplace=True here
df = data.drop('Loan_ID', axis=1)

def to_numeric(df):
    to_numeric = {'Male': 1, 'Female': 2,
    'Yes': 1, 'No': 2,
    'Graduate': 1, 'Not Graduate': 2,
    'Urban': 3, 'Semiurban': 2, 'Rural': 1,
    'Y': 1, 'N': 0,
    '3+': 3}

    # No need for inplace=True here
    data = df.applymap(lambda label: to_numeric.get(label) if label in to_numeric else label)

    return data

df_num = to_numeric(df)

def fill_data_mode(df):
    # Fill up data with mode
    null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
    for col in null_cols:
        df[col] = df[col].fillna(df[col].dropna().mode().values[0])
    return df

# Streamlit app display
st.title("Loan Status Prediction Comparison")

# Load and preprocess data
st.write("Fill Data with Mode")
df_num_mode = fill_data_mode(to_numeric(df))

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

# Define models with best hyperparameters
models = {
    'Logistic Regression': LogisticRegression(random_state=start_state),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=start_state),
    'k-Nearest Neighbor': KNeighborsClassifier(n_neighbors=9),
    'Decision Tree': DecisionTreeClassifier(max_depth=1, random_state=start_state)
}

# Dictionary to store maximum accuracy and cross-validation score for each model
max_accuracies = {}
mean_cv_scores = {}

# Evaluate each model
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    max_accuracies[model_name] = test_score

    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = np.mean(cv_scores)
    mean_cv_scores[model_name] = mean_cv_score

# Bar plot for maximum accuracies
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Plot Maximum Accuracy
ax[0].bar(max_accuracies.keys(), max_accuracies.values(), color=['blue', 'green', 'orange', 'red'])
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Maximum Accuracy Comparison')

# Plot Cross-Validation Scores
ax[1].bar(mean_cv_scores.keys(), mean_cv_scores.values(), color=['blue', 'green', 'orange', 'red'])
ax[1].set_ylabel('Cross-Validation Score')
ax[1].set_title('Cross-Validation Score Comparison')

st.pyplot(fig)

# Display detailed results for each model
st.subheader("Model Comparison Details:")
for model_name, model in models.items():
    st.write(f"### {model_name}")
    
    # Cross-validation scores
    mean_cv_score = mean_cv_scores[model_name]
    st.write(f"Mean Cross-validation Score: {mean_cv_score:.2%}")
    
    # Confusion Matrix
    y_pred = model.predict(X_test_scaled)
    conf_mat = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_mat)
    ConfusionMatrixDisplay(conf_mat, display_labels=['Not Approved','Approved']).plot()
    confusion_fig = plt.gcf()
    st.pyplot(confusion_fig)

    # Classification Report
    classification_report_str = classification_report(y_test, y_pred)
    st.write("Classification Report:")
    st.text(classification_report_str)
    st.write("---")
