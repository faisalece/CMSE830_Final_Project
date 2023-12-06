import streamlit as st

def display_reference_page():
    st.markdown("## Loan Status and Reference:")
    st.write("When individuals apply for a loan from a bank, several key parameters are considered to assess their creditworthiness and determine the eligibility for the loan. Banks evaluate the applicant's financial stability, credit history, income, and debt levels. Additionally, factors such as employment history, loan purpose, and the applicant's relationship with the bank may also play a crucial role in the decision-making process. The overall goal is to gauge the borrower's ability to repay the loan in a timely manner. These parameters collectively help banks assess the risk associated with lending to a particular individual, ensuring responsible lending practices.")

    st.image("loan_give.jpeg", caption="Here is your loan amount.", use_column_width=True)

    st.markdown('[Source : Kaggle Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)')

    st.markdown("## Dataset:")
    st.write("The Loan Prediction Problem Dataset is sourced from Kaggle and serves as the foundation for this machine learning project. It contains crucial information about loan applicants, including various features such as gender, education, income, and property area. The target variable, 'Loan_Status,' represents whether a loan was approved or denied. The dataset is instrumental in training and evaluating machine learning models to predict loan approvals.")

    st.image("loan.jpg", caption="May I get a loan?", use_column_width=True)

    st.write("Explore the dataset and its features to gain insights into the factors influencing loan approval decisions.")

if __name__ == "__main__":
    display_reference_page()
