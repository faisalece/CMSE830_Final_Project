import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, accuracy_score

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


def fill_data_median(df):
    #Fill up data with median
    df['Credit_History'].fillna(value=df['Credit_History'].median(),inplace=True)
    df['Self_Employed'].fillna(value=df['Self_Employed'].median(),inplace=True)
    df['LoanAmount'].fillna(value=df['LoanAmount'].median(),inplace=True)
    df['Dependents'].fillna(value=df['Dependents'].median(),inplace=True)
    df['Loan_Amount_Term'].fillna(value=df['Loan_Amount_Term'].median(),inplace=True)
    df['Gender'].fillna(value=df['Gender'].median(),inplace=True)
    df['Married'].fillna(value=df['Married'].median(),inplace=True)
    return df
        
def fill_data_KNN(df):
    #Fill up data with KNN
    my_imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
    df_repaired = pd.DataFrame(my_imputer.fit_transform(df), columns=df.columns)
    return df_repaired

def fill_data_mode(df):
    #Fill up data with mode
    null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
    for col in null_cols:
        df[col] = df[col].fillna(
        df[col].dropna().mode().values[0] )  
    return df


#st.write(df_numerical.head())

    
# Streamlit app display
st.title("Loan Status Prediction Using Logistic Regression")
    

#tabs
mode,median,KNN= st.tabs(["Fill Data By Mode", "Fill Data By Median", "Fill Data By KNN"])

with mode:
    st.write("Fill Data with Mode")
    df_num_mode = fill_data_mode(df_num)
    # Split the data into features (X) and target variable (y)
    y1 = df_num_mode['Loan_Status']
    X1 = df_num_mode.drop('Loan_Status', axis=1)
    
    # Split the data into training and testing sets
    start_state = 42
    test_fraction = 0.2
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=test_fraction, random_state=start_state)
    #st.write(X_train1, X_test1, y_train1, y_test1)
    
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled1 = scaler.fit_transform(X_train1)
    X_test_scaled1 = scaler.transform(X_test1)
    
    # Create and train the Logistic Regression classifier
    lr_classifier1 = LogisticRegression()
    lr_model1 = lr_classifier1.fit(X_train_scaled1, y_train1)
    
    # Evaluate the model on the test set
    test_score1 = lr_model1.score(X_test_scaled1, y_test1)
    
    # Perform cross-validation
    cv_scores1 = cross_val_score(lr_classifier1, X1, y1, cv=5)
    
    # Make predictions on the test set
    y_pred1 = lr_model1.predict(X_test_scaled1)

    # Display test set score
    st.write(f"The accuracy of the model on the test set is {test_score1:.2%}")
    
    # Display cross-validation scores
    mean_cv_score1 = np.mean(cv_scores1)
    st.write(f"The mean cross-validation score is: {mean_cv_score1}")
    
    # Display the confusion matrix
    st.write("Confusion Matrix:")
    #ConfusionMatrixDisplay.from_estimator(lr_classifier1, X_test_scaled1, y_test1)
    conf_mat1 = confusion_matrix(y_test1, y_pred1)
    ConfusionMatrixDisplay(conf_mat1, display_labels=['Not Approved','Approved']).plot()
    confusion_mean_fill_fig1 = plt.gcf()  # Get the current figure
    st.pyplot(confusion_mean_fill_fig1)
    
    # Prediction Summary by Species
    st.write("Prediction Summary by Species:")
    classification_report_str1 = classification_report(y_test1, y_pred1)
    st.text(classification_report_str1)


with median:
    st.write("Fill Data with Median")
    df_num_median = fill_data_median(df_num)
    # Split the data into features (X) and target variable (y)
    y = df_num_median['Loan_Status']
    X = df_num_median.drop('Loan_Status', axis=1)
    
    # Split the data into training and testing sets
    start_state = 42
    test_fraction = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)
    #st.write(X_train, X_test, y_train, y_test)
    
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the Logistic Regression classifier
    lr_classifier = LogisticRegression()
    lr_model = lr_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate the model on the test set
    test_score = lr_model.score(X_test_scaled, y_test)
    
    # Perform cross-validation
    cv_scores = cross_val_score(lr_classifier, X, y, cv=5)
    
    # Make predictions on the test set
    y_pred = lr_model.predict(X_test_scaled)
    
    # Generate and display the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Display test set score
    st.write(f"The accuracy of the model on the test set is {test_score:.2%}")
    
    # Display cross-validation scores
    mean_cv_score_median = np.mean(cv_scores)
    st.write(f"The mean cross-validation score is: {mean_cv_score_median}")
    
    # Display the confusion matrix
    st.write("Confusion Matrix:")
    #ConfusionMatrixDisplay.from_estimator(lr_classifier, X_test_scaled, y_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(conf_mat, display_labels=['Not Approved','Approved']).plot()
    confusion_mean_fill_fig = plt.gcf()  # Get the current figure
    st.pyplot(confusion_mean_fill_fig)
    
    # Prediction Summary by Species
    st.write("Prediction Summary by Species:")
    classification_report_str = classification_report(y_test, y_pred)
    st.text(classification_report_str)

with KNN:
    st.write("Fill Data with KNN")
    df_num_KNN = fill_data_KNN(df_num)
    # Split the data into features (X) and target variable (y)
    y = df_num_KNN['Loan_Status']
    X = df_num_KNN.drop('Loan_Status', axis=1)
    
    # Split the data into training and testing sets
    start_state = 42
    test_fraction = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)
    #st.write(X_train, X_test, y_train, y_test)
    
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the Logistic Regression classifier
    lr_classifier = LogisticRegression()
    lr_model = lr_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate the model on the test set
    test_score = lr_model.score(X_test_scaled, y_test)
    
    # Perform cross-validation
    cv_scores = cross_val_score(lr_classifier, X, y, cv=5)
    
    # Make predictions on the test set
    y_pred = lr_model.predict(X_test_scaled)
    
    # Generate and display the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Display test set score
    st.write(f"The accuracy of the model on the test set is {test_score:.2%}")
    
    # Display cross-validation scores
    mean_cv_score_median = np.mean(cv_scores)
    st.write(f"The mean cross-validation score is: {mean_cv_score_median}")
    
    # Display the confusion matrix
    st.write("Confusion Matrix:")
    #ConfusionMatrixDisplay.from_estimator(lr_classifier, X_test_scaled, y_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(conf_mat, display_labels=['Not Approved','Approved']).plot()
    confusion_mean_fill_fig = plt.gcf()  # Get the current figure
    st.pyplot(confusion_mean_fill_fig)
    
    # Prediction Summary by Species
    st.write("Prediction Summary by Species:")
    classification_report_str = classification_report(y_test, y_pred)
    st.text(classification_report_str)


