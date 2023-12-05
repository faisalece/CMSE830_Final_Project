import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

#df = to_numeric(df)

#tabs
gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area = st.tabs(["Gender", "Married", "Dependents","Education","Self Employed", "Applicant Income", "Co-applicant Income", "Loan Amount", "Loan Amount Term", "Credit History", "Property Area"])

with gender:
    # Gender
    plt.figure(figsize=(15,5))
    sns.countplot(x='Gender', hue='Loan_Status', data=df);
    gender_fig = plt.gcf()  # Get the current figure
    st.pyplot(gender_fig) 
    import streamlit as st
    st.write("Most males and females have obtained loans, indicating no clear pattern.")
    st.write("Decision: Not an important feature")

with married:
    # married
    plt.figure(figsize=(15,5))
    sns.countplot(x='Married', hue='Loan_Status', data=df);
    married_fig = plt.gcf()  # Get the current figure
    st.pyplot(married_fig)
    st.write("Most people who get married have obtained a loan.")
    st.write("If you're married, then you may have a better chance of getting a loan.")
    st.write("##### Decision: Good feature")
    
with dependents:
    #dependents
    plt.figure(figsize=(15,5))
    sns.countplot(x='Dependents', hue='Loan_Status', data=df);
    dependents_fig = plt.gcf()  # Get the current figure
    st.pyplot(dependents_fig)
    st.write("If Dependents = 0, there is a higher chance of getting a loan (very high chance).")
    st.write("##### Decision: Good feature")

with education:
    # education
    plt.figure(figsize=(15,5))
    sns.countplot(x='Education', hue='Loan_Status', data=df);
    education_fig = plt.gcf()  # Get the current figure
    st.pyplot(education_fig)
    st.write("Whether you are graduated or not, you will have almost the same chance of getting a loan (No clear pattern).")
    st.write("Observation: Most people are graduates, and most of them have obtained a loan.")
    st.write("However, people who didn't graduate also got a loan, but with a lower percentage compared to those who graduated.")
    st.write("##### Decision: Not an important feature")

with self_employed:
    # self_employed
    plt.figure(figsize=(15,5))
    sns.countplot(x='Self_Employed', hue='Loan_Status', data=df);
    Self_Employed_fig = plt.gcf()  # Get the current figure
    st.pyplot(Self_Employed_fig)
    st.write("##### Decision: No pattern (same as Education)")

with applicant_income:
    # ApplicantIncome
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x='ApplicantIncome', y='Loan_Status', data=df)
    plt.title("ApplicantIncome vs Loan_Status")
    plt.xlabel("ApplicantIncome")
    plt.ylabel("Loan_Status")
    applicant_income_fig = plt.gcf()
    st.pyplot(applicant_income_fig)
    st.write("##### Decision: No clear pattern observed.")

with coapplicant_income:
    #coapplicant_income
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x='CoapplicantIncome', y='Loan_Status', data=df)
    plt.title("CoapplicantIncome vs Loan_Status")
    plt.xlabel("CoapplicantIncome")
    plt.ylabel("Loan_Status")
    applicant_income_fig = plt.gcf()
    st.pyplot(applicant_income_fig)
    st.write("##### Decision: No clear pattern observed.")

with loan_amount:
    #loan_amount
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x='LoanAmount', y='Loan_Status', data=df)
    plt.title("Loan Amount vs Loan_Status")
    plt.xlabel("Loan Amount")
    plt.ylabel("Loan_Status")
    loan_amount_fig = plt.gcf()
    st.pyplot(loan_amount_fig)
    st.write("##### Decision: No clear pattern observed.")
    
with loan_amount_term:
    #loan_amount_term
    plt.figure(figsize=(15, 5))
    sns.scatterplot(x='Loan_Amount_Term', y='Loan_Status', data=df)
    plt.title("Loan Amount Term vs Loan_Status")
    plt.xlabel("Loan Amount Term")
    plt.ylabel("Loan_Status")
    loan_amount_term_fig = plt.gcf()
    st.pyplot(loan_amount_term_fig)
    st.write("##### Decision: No clear pattern observed.")

with credit_history:
    # credit_history
    plt.figure(figsize=(15,5))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df);
    credit_history_fig = plt.gcf()  # Get the current figure
    st.pyplot(credit_history_fig)
    st.write("We didn't approve loans for most people with Credit History = 0.")
    st.write("However, we approved loans for most people with Credit History = 1.")
    st.write("Conclusion: If you have Credit History = 1, you will have a better chance of getting a loan.")
    st.write("##### Decision: Important feature")

with property_area:
    #Property_Area
    plt.figure(figsize=(15,5))
    sns.countplot(x='Property_Area', hue='Loan_Status', data=df);
    property_area_fig = plt.gcf()  # Get the current figure
    st.pyplot(property_area_fig)
    st.write("Residents of Semi-Urban areas are more likely to secure loans compared to those in urban or rural areas.")
    st.write("##### Decision: Important feature")
    
df_new = to_numeric(df)
st.write("### Correlation Heatmap:")
#plt.figure(figsize=(10, 10))
sns.heatmap(df_new.corr(), annot=True, cmap='coolwarm')
heatmap_fig = plt.gcf()  # Get the current figure
st.pyplot(heatmap_fig)

# Distribution of Loan Amount
st.subheader("Distribution of Loan Amount:")
fig_loan_amount, ax_loan_amount = plt.subplots()
sns.histplot(data['LoanAmount'], kde=True, bins=20, color='skyblue')
st.pyplot(fig_loan_amount)

# Boxplot of Applicant Income by Education
st.subheader("Applicant Income Distribution by Education:")
fig_income_boxplot, ax_income_boxplot = plt.subplots()
sns.boxplot(x='Education', y='ApplicantIncome', data=data, palette='pastel')
st.pyplot(fig_income_boxplot)

# Count plot of Loan Approval Status
st.subheader("Loan Approval Status Count:")
fig_loan_status_count, ax_loan_status_count = plt.subplots()
sns.countplot(x='Loan_Status', data=data, palette='Set2')
st.pyplot(fig_loan_status_count)

