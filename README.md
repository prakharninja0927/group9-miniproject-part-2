# group9-miniproject-part-2

## Loan Approval Classification Using Neural Netwrok and Streamlit ["streamlit- link"](https://group9-miniproject-part-2-u0cmy79oacn.streamlit.app/)

1. The necessary libraries are imported: `streamlit`, `pandas`, `tensorflow`, `matplotlib.pyplot`, `train_test_split` from `sklearn.model_selection`, `MinMaxScaler` and `LabelEncoder` from `sklearn.preprocessing`, `os`, and `load_model` from `tensorflow.keras.models`.

2. The `load_data` function is defined to load the loan dataset from a CSV file. It uses the `pd.read_csv` function to read the data from a URL and returns the loaded data.

3. The `preprocessed_input_data` function is defined to preprocess the input data. It handles missing values by filling them with the mode (most frequent value) of each feature. It also applies label encoding to categorical features to convert them into numerical values. Additionally, it cleans the `Dependents` feature by keeping only the first character. The function returns the preprocessed data.

4. The `preprocess_data` function is defined to preprocess the entire dataset. It drops the `Loan_ID` column, handles missing values, applies label encoding, and cleans the `Dependents` feature. It separates the input features `X` and the target variable `y` and returns them.

5. The `build_model` function is defined to build and train a neural network model. It uses the Sequential API from TensorFlow's Keras to define a model with multiple dense layers. It compiles the model with the Adam optimizer, binary cross-entropy loss function, and accuracy metric. It then fits the model on the training data for a specified number of epochs and batch size. Finally, it returns the trained model.

6. The `main` function is defined as the Streamlit application's entry point. It starts by displaying the headers and subheaders using `st.header` and `st.subheader`.

7. Next, the user can select the number of samples to generate by using a slider. The `load_data` function is called with the selected number of samples, and the resulting data is displayed using `st.dataframe`.

8. The data is then preprocessed by calling the `preprocess_data` function, splitting it into training and testing sets using `train_test_split` from `sklearn.model_selection`.

9. The code checks if a pre-trained model file exists. If it does, the model is loaded using `load_model` from `tensorflow.keras.models`. Otherwise, a new model is built and trained by calling the `build_model` function. The trained model is then saved to a file using `model.save`.

10. The model's performance is evaluated on the test data by calling `model.evaluate`, and the loss and accuracy are displayed using `st.write`.

11. The user can make loan approval predictions by selecting various input features such as gender, marital status, number of dependents, education level, employment status, income, loan amount, loan term, credit history, and property area using Streamlit's input components.

12. When the user clicks the "Predict" button, the selected input values are used to create a new row of data. The data is preprocessed using `preprocessed_input_data`, and the preprocessed input is passed to the trained model for prediction using `model.predict`. The loan approval prediction is displayed as either "Loan Approved" or "Loan Rejected" using colored text.

13. Finally, the model summary is displayed in an expander section using `model.summary` and Streamlit's `st.expander`.

In summary, the provided code is a Streamlit application that loads loan data, preprocesses it, builds and trains a neural network model, and allows users to make loan approval predictions based on selected input features. The application also provides an evaluation of the model's performance and displays a summary of the model architecture.


## Following are some screen shots of our working app.
![s1](https://github.com/prakharninja0927/group9-miniproject-part-2/assets/70143550/5ea6f7b0-fdbc-4d53-93b4-2ad33a2c81e1)
![s2](https://github.com/prakharninja0927/group9-miniproject-part-2/assets/70143550/5c8ee1aa-dd77-4a39-8dbd-7d7c48aa5136)
![s3](https://github.com/prakharninja0927/group9-miniproject-part-2/assets/70143550/d16389d4-945f-4dd9-a521-07a571f59b22)
