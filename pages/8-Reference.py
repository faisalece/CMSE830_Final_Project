import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    data = pd.read_csv('train.csv')
    return data

data = load_data()

# Set page title and icon
st.set_page_config(
    page_title="Loan Status Analysis",
    page_icon="💰",
    layout="wide"
)

# Display source link
st.markdown('[Source: Kaggle Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)')

# Add a slider for selecting the number of rows to display
num_rows = st.slider("Number of Rows to Display", 1, len(data), 100)

# Display the selected number of rows
st.write(f"Displaying top {num_rows} rows:")
st.write(data.head(num_rows))

# Add any other visualizations or analysis as needed
# ...

# Example: Distribution of Loan Amount
st.subheader("Distribution of Loan Amount:")
column = st.selectbox("Select a column", data.columns, index=data.columns.get_loc('LoanAmount'))
bins = st.slider("Number of bins", 5, 100, 50)
st.write("Histogram:")
fig, ax = plt.subplots()
sns.histplot(data=data, x=column, hue="Loan_Status", bins=bins, kde=True)
plt.xlabel(column)
plt.ylabel("Count")
plt.title(f"Distribution of {column} by Loan Status")
st.pyplot(fig)
