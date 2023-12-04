import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
st.title("Loan Status Prediction Using K-Nearest Neighbors")

st.write("Fill Data with Mode")
df_num_mode = fill_data_mode(df_num)

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

# Create a range of k values to try
k_values = list(range(1, 21))

# Initialize variables to store best parameters and accuracy
best_k = None
best_accuracy = 0.0

# Perform grid search
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_model = knn_classifier.fit(X_train_scaled, y_train)
    test_score = knn_model.score(X_test_scaled, y_test)

    # Update best parameters if the current model is better
    if test_score > best_accuracy:
        best_accuracy = test_score
        best_k = k

# Display the best k and corresponding accuracy
st.write(f"The best k is {best_k} with an accuracy of {best_accuracy:.2%}")

k_value = st.slider("Number of Neighbors (k)", 1, 20, 5)
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
knn_model = knn_classifier.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
current_test_score = knn_model.score(X_test_scaled, y_test)
# Perform cross-validation
cv_scores = cross_val_score(knn_classifier, X, y, cv=5)

# Make predictions on the test set
y_pred = knn_model.predict(X_test_scaled)

# Generate and display the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Display test set score
st.write(f"The accuracy of the model on the test set is {current_test_score:.2%} for k = {k_value}")

# Display cross-validation scores
mean_cv_score_mode = np.mean(cv_scores)
st.write(f"The mean cross-validation score is: {mean_cv_score_mode}")

# Display the confusion matrix
st.write("Confusion Matrix:")
conf_mat = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_mat, display_labels=['Not Approved','Approved']).plot()
confusion_mean_fill_fig = plt.gcf()  # Get the current figure
st.pyplot(confusion_mean_fill_fig)

# Prediction Summary by Species
st.write("Prediction Summary by Species:")
classification_report_str = classification_report(y_test, y_pred)
st.text(classification_report_str)
