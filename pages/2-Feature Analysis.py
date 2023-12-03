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
    st.write("Most males have obtained a loan, and most females have obtained one too, indicating no clear pattern.")
    st.write("I think it's not a very important feature; we will reassess its significance later.")
    st.write("##### Decision: To be reevaluated")

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
    
df_new = to_numeric(df)
st.write("### Correlation Heatmap:")
#plt.figure(figsize=(10, 10))
sns.heatmap(df_new.corr(), annot=True, cmap='coolwarm')
heatmap_fig = plt.gcf()  # Get the current figure
st.pyplot(heatmap_fig)
