import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
from tensorflow.keras.models import load_model

# Generate sample data
def load_data(num_samples):
    loan_data  = pd.read_csv("https://raw.githubusercontent.com/prakharninja0927/Loan-Approval-Classification/main/Data/Training%20Dataset.csv" )
    return loan_data


# Define the Streamlit app
def main():
    st.header("Loan Approval Prediction")
    st.subheader("Generate Sample Data")

    # Generate and display sample data
    num_samples = st.slider("Number of Samples", min_value=100, max_value=1000, step=100)
    data = load_data(num_samples)
    st.dataframe(data)  

if __name__ == '__main__':
    main()
