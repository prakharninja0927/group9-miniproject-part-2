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

def preprocessed_input_data(data):
    ### Applying Label Encoding to categorical features for better result
     ### Applying Label Encoding to categorical features for better result

    # Filling null values with mode() of that particular features
    data["Gender"].fillna(value=data['Gender'].mode()[0],inplace=True)
    data["Married"].fillna(value=data['Married'].mode()[0],inplace=True)
    data["Dependents"].fillna(value=data['Dependents'].mode()[0],inplace=True)
    data["Self_Employed"].fillna(value=data['Self_Employed'].mode()[0],inplace=True)
    data["LoanAmount"].fillna(value=data['LoanAmount'].mode()[0],inplace=True)
    data["Loan_Amount_Term"].fillna(value=data['Loan_Amount_Term'].mode()[0],inplace=True)
    data["Credit_History"].fillna(value=data['Credit_History'].mode()[0],inplace=True)

    label_encoder = LabelEncoder()

    data['Property_Area']= label_encoder.fit_transform(data['Property_Area']) 
    data['Gender']= label_encoder.fit_transform(data['Gender'])
    data['Married']= label_encoder.fit_transform(data['Married'])
    data['Education']= label_encoder.fit_transform(data['Education']) 
    data['Self_Employed']= label_encoder.fit_transform(data['Self_Employed'])

    def clean_dep(x):  
        return x[0]
    data['Dependents'] = data['Dependents'].astype(str).apply(clean_dep)


    return data.astype('float32')

# Preprocess the data
def preprocess_data(data):

    # We dont need Loan_ID column so we are going to drop it
    data = data.drop(columns=['Loan_ID'])

    # Filling null values with mode() of that particular features
    data["Gender"].fillna(value=data['Gender'].mode()[0],inplace=True)
    data["Married"].fillna(value=data['Married'].mode()[0],inplace=True)
    data["Dependents"].fillna(value=data['Dependents'].mode()[0],inplace=True)
    data["Self_Employed"].fillna(value=data['Self_Employed'].mode()[0],inplace=True)
    data["LoanAmount"].fillna(value=data['LoanAmount'].mode()[0],inplace=True)
    data["Loan_Amount_Term"].fillna(value=data['Loan_Amount_Term'].mode()[0],inplace=True)
    data["Credit_History"].fillna(value=data['Credit_History'].mode()[0],inplace=True)

    ### Applying Label Encoding to categorical features for better result
    label_encoder = LabelEncoder()

    data['Property_Area']= label_encoder.fit_transform(data['Property_Area']) 
    data['Gender']= label_encoder.fit_transform(data['Gender'])
    data['Married']= label_encoder.fit_transform(data['Married'])
    data['Education']= label_encoder.fit_transform(data['Education']) 
    data['Self_Employed']= label_encoder.fit_transform(data['Self_Employed'])
    data['Loan_Status']= label_encoder.fit_transform(data['Loan_Status'])

    def clean_dep(x):
        return x[0]
    data['Dependents'] = data['Dependents'].apply(clean_dep)
    # data.Dependents_Clean = data.Dependents.astype(dtype='int8')
    data = pd.DataFrame(data)

    X = data.iloc[:, :-1].astype('float32')
    y = data.iloc[:, -1].astype('float32')
    
    return X, y

# Build and train the neural network model
def build_model(X_train, y_train):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16)

    return model

# Define the Streamlit app
def main():
    st.header("Loan Approval Prediction")
    st.subheader("Generate Sample Data")

    # Generate and display sample data
    num_samples = st.slider("Number of Samples", min_value=100, max_value=1000, step=100)
    data = load_data(num_samples)
    st.dataframe(data)  

     # Preprocess the data
    X, y = preprocess_data(data)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Load Model or Train the model
    st.subheader("Train the Model")

    # Specify the file name for saving the model
    model_filename = "loan_model.h5"

    # Check if the file already exists
    if os.path.exists(model_filename):
        # Load the existing model
        model = load_model(model_filename)
        print("Existing model loaded.")
    else:
        # Train and save the model
        model = build_model(X_train, y_train)
        model.save(model_filename)
        print("New model trained and saved.")

    # Evaluate the model
    st.subheader("Evaluate the Model")
    loss, accuracy = model.evaluate(X_test, y_test)
    st.write("Loss:", loss)
    st.write("Accuracy:", accuracy)

    # Make predictions
    st.subheader("Make Predictions")
    gender = st.selectbox("Select Gender",['Male','Female'])
    married = st.selectbox("Are you Married?",['Yes','No'])
    dependents = st.slider("Number of Dependents",min_value=0, max_value=10, step=1)
    education = st.radio('Education Level',['Graduate','Not Graduate'], index=1)
    self_employed = st.radio('Self Employed?',['Yes','No'])
    applicant_income = st.number_input('Enter your Income')
    coapplicant_income = st.number_input('Enter Co-applicantIncome Income')
    loan_amount = st.number_input("Enter Loan Amount")
    loan_amount_term = st.number_input("Enter Loan Term in Days")
    credit_history = st.selectbox("Credit History", [0, 1])
    property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])



    # st.write(prediction)
    if st.button('Predict'):
        
        new_row = {
        'Gender': gender,
        'Married': married,
        'Dependents': str(int(dependents)),
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': float(applicant_income),
        'CoapplicantIncome': float(coapplicant_income),
        'LoanAmount': float(loan_amount),
        'Loan_Amount_Term': int(loan_amount_term),
        'Credit_History': float(credit_history),
        'Property_Area': property_area
         }
        # st.dataframe(input_data.dtypes)
        df = data.drop(columns=['Loan_ID','Loan_Status'])

        df2 = df.append(new_row,ignore_index = True)

        in_data = preprocessed_input_data(df2)

        # Make loan approval prediction
        prediction = model.predict(in_data.tail(1))

        if prediction > 0.5:
            st.markdown('<p style="color:green;font-size:30px;">Loan Approved</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:Red;font-size:30px;"><b>Loan Rejected</b></p>', unsafe_allow_html=True)

        prediction=0
    # Display the model summary using Streamlit
    st.write('## Model Summary')
    with st.expander("Click here to see the model summary"):
        model.summary(print_fn=st.write)


if __name__ == '__main__':
    main()
