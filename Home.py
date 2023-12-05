import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()
df = data

# Set page title and icon
st.set_page_config(
    page_title="Loan Status Analysis",
    page_icon="💰",
    layout="wide"
)

# Header
st.title("Loan Status Analysis!")
col1, col2 = st.columns([2, 5])
with col1:
    st.image("loan.jpg", caption="May I get a loan?", use_column_width=True)
with col2:
    st.write("In finance, a loan is when someone lends money to another person or organization. The person or organization receiving the money, known as the borrower, takes on a debt. This means they have to pay back the borrowed amount, plus an extra charge called interest. The borrower continues to pay until the entire borrowed amount is repaid. Loans serve as a common way for individuals and organizations to access financial support for different needs, making money more accessible for various purposes.")
        


#tabs
intro_tab, goal_tab, describe_tab, significance_tab, con_tab = st.tabs(["Introduction", "Project Goal", "Describe the Dataset","Project Significance","Conclusion"])

with intro_tab:
    col3, col4 = st.columns([3, 2])
    with col3:
        st.write("When individuals apply for a loan from a bank, several key parameters are considered to assess their creditworthiness and determine the eligibility for the loan. Banks evaluate the applicant's financial stability, credit history, income, and debt levels. Additionally, factors such as employment history, loan purpose, and the applicant's relationship with the bank may also play a crucial role in the decision-making process. The overall goal is to gauge the borrower's ability to repay the loan in a timely manner. These parameters collectively help banks assess the risk associated with lending to a particular individual, ensuring responsible lending practices.")
        st.write("Here are someparameters that banks commonly consider when evaluating loan applications:")
        st.write("1. Credit Score")
        st.write("2. Income")
        st.write("3. Debt-to-Income Ratio")
        st.write("4. Employment History")
        st.write("5. Loan Purpose")
        st.write("6. Family Size")
        st.write("Etc.")
    with col4:
        st.image("loan_give.jpeg", caption="Here is your loan amount.", use_column_width=True)

    st.markdown('[Source : Kaggle Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)')
        # Add a slider for selecting the number of rows to display
        num_rows = st.slider("Number of Rows", 1, 600, 100)

        # Display the selected number of rows
        st.write(f"Displaying top {num_rows} rows:")
        st.write(data.head(num_rows))
    with goal_tab:
        st.write("The main objective of this mid-term project is to conduct a thorough analysis of the Water Quality dataset in order to assess the safety of water sources for consumption. Specifically, our aim is to develop a predictive model that can accurately determine the drinkability of water based on various comprehensive water quality parameters.")   
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader('| SUMMARY')
            col = len(data.columns)-1
            st.write('PARAMETERS : ',col)
            row = len(data) 
            st.write('TOTAL DATA : ', row)
            st.write("Potability Distribution (Pie Chart)")
            potability_counts = data['Loan_Status'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(potability_counts, labels=potability_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
        with col2:
            st.write("This research aims to determine if a comprehensive analysis of water quality parameters can accurately predict the drinkability of water sources. Additionally, we seek to understand how the findings from this analysis can contribute to addressing the critical concern of ensuring safe drinking water for everyone. The significance of this project lies in its potential to have a direct impact on public health and well-being. Access to clean and safe drinking water is a basic human right, and by conducting this analysis, we hope to provide valuable insights that can inform water management decisions and help ensure the provision of safe drinking water to communities in need.")

    with describe_tab:
        st.write(data.describe())

