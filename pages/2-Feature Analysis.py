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
    st.write("Decision: To be reevaluated")

with married:
    # married
    plt.figure(figsize=(15,5))
    sns.countplot(x='Married', hue='Loan_Status', data=df);
    married_fig = plt.gcf()  # Get the current figure
    st.pyplot(married_fig)
    st.write("Most people who get married have obtained a loan.")
    st.write("If you're married, then you may have a better chance of getting a loan.")
    st.write("##### Decision: Good feature")

plot_options = ["Correlation Heat Map", "Joint Plot of Columns","Histogram of Column", "Pair Plot", "PairGrid Plot", "Box Plot of Column", "3D Scatter Plot"]
selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

if selected_plot == "Correlation Heat Map":
    st.write("Correlation Heatmap:")
    #plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    heatmap_fig = plt.gcf()  # Get the current figure
    st.pyplot(heatmap_fig)

elif selected_plot == "Joint Plot of Columns":
    x_axis = st.sidebar.selectbox("Select x-axis", df.columns, index=0)
    y_axis = st.sidebar.selectbox("Select y-axis", df.columns, index=1)
    st.write("Joint Plot:")
    jointplot = sns.jointplot(data = df, x=df[x_axis], y=df[y_axis], hue="Loan_Status")
    #sns.scatterplot(data = df, x=df[x_axis], y=df[y_axis], hue="Potability", ax=ax)
    st.pyplot(jointplot)

elif selected_plot == "Histogram of Column":
    column = st.sidebar.selectbox("Select a column", df.columns)
    bins = st.sidebar.slider("Number of bins", 5, 100, 20)
    st.write("Histogram:")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, hue="Loan_Status",bins=bins, kde=True)
    st.pyplot(fig)

elif selected_plot == "Pair Plot":
    st.subheader("Pair Plot")
    selected_box = st.multiselect('Select variables:', [col for col in df.columns if col != 'Loan_Status'])
    selected_data = df[selected_box + ['Loan_Status']]  # Add 'Potability' column
    all_columns = selected_data.columns
    exclude_column = 'Loan_Status' 
    dims = [col for col in all_columns if col != exclude_column]
    fig = px.scatter_matrix(selected_data, dimensions=dims, title="Pair Plot", color='Loan_Status')
    fig.update_layout(plot_bgcolor="white")  
    st.plotly_chart(fig)

elif selected_plot == "PairGrid Plot":
    st.subheader("Pair Plot")
    selected_box = st.multiselect('Select variables:', [col for col in df.columns if col != 'Loan_Status'],default=['ph'])
    selected_data = df[selected_box + ['Loan_Status']]  # Add 'Potability' column

    # Create a PairGrid
    g = sns.PairGrid(selected_data, hue='Loan_Status')
    g.map_upper(plt.scatter)
    g.map_diag(plt.hist, histtype="step", linewidth=2, bins=30)
    g.map_lower(plt.scatter)
    g.add_legend()

    # Display the PairGrid plot
    st.pyplot(plt.gcf())

elif selected_plot == "Box Plot of Column":
    column = st.sidebar.selectbox("Select a column", df.columns)
    st.write("Box Plot:")
    fig, ax = plt.subplots()
    sns.boxplot(df[column], ax=ax)
    st.pyplot(fig)

elif selected_plot == "3D Scatter Plot":
    x_axis = st.sidebar.selectbox("Select x-axis", df.columns, index=0)
    y_axis = st.sidebar.selectbox("Select y-axis", df.columns, index=1)
    z_axis = st.sidebar.selectbox("Select z-axis", df.columns, index=2)
    st.subheader("3D Scatter Plot")
    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color='Potability')
    st.plotly_chart(fig)
