cimport streamlit as st
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
    with col4:
        st.image("loan_give.jpeg", caption="Here is your loan amount.", use_column_width=True)

    st.markdown('[Source : Kaggle Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)')
    # Add a slider for selecting the number of rows to display
    num_rows = st.slider("Number of Rows", 1, 600, 100)

    # Display the selected number of rows
    st.write(f"Displaying top {num_rows} rows:")
    st.write(data.head(num_rows))
with goal_tab:
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
        st.write("The project goal for the Loan Prediction Problem Dataset is to develop a robust machine learning model that accurately predicts whether a loan application should be approved or denied based on relevant features. The objective is to enhance the efficiency of the lending process by automating decision-making while minimizing the risk of default. Through analysis and prediction, the project aims to contribute to the optimization of loan approval procedures and facilitate more informed lending decisions.")

with describe_tab:
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

    data = to_numeric(df)
    st.write(data.describe())
    # Display basic statistics
    st.subheader("Dataset Overview:")
    st.write(data.info())

    # Display missing values
    st.subheader("Missing Values:")
    st.write(data.isnull().sum())

    # Distribution of Loan Amount
    st.subheader("Distribution of Loan Amount:")
    ax_loan_amount = plt.subplots()
    sns.histplot(data['LoanAmount'], kde=True, bins=20, color='skyblue')
    loan_amount_fig = plt.gcf()
    st.pyplot(loan_amount_fig)

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
    
with significance_tab:
    st.write("The project holds significant importance as it evaluates and compares multiple machine learning algorithms, namely Logistic Regression, Random Forest, K-Nearest Neighbors, and Decision Tree, for predicting loan approvals. By assessing the performance of these models, the project aims to identify the most effective approach for loan prediction, providing valuable insights to financial institutions. The findings contribute to the optimization of lending practices, enhancing decision-making processes, and potentially reducing the risk of defaults. This comparative analysis aids in selecting the most suitable algorithm for accurate and reliable loan approval predictions, fostering efficiency and trust in the lending industry.")

with con_tab:
    st.write("In conclusion, the comprehensive evaluation of Logistic Regression, Random Forest, K-Nearest Neighbors, and Decision Tree models for the Loan Prediction Problem Dataset has provided valuable insights into their respective performances. Through meticulous comparison, it was observed that [mention the best-performing model], demonstrating superior accuracy and reliability in predicting loan approvals. This finding is crucial for financial institutions seeking to enhance their decision-making processes, streamline loan approval procedures, and mitigate risks associated with defaults. The project's outcomes not only contribute to the field of machine learning but also have practical implications, guiding the adoption of the most effective algorithm for optimizing the lending industry's efficiency and efficacy.")
