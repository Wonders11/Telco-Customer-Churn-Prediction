import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler
with open('label_encoders.pkl','rb') as file:
    label_encoders = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot_encoder = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Telco Customer Churn Prediction')

# Streamlit UI for user inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
Partner = st.selectbox('Partner', ['No', 'Yes'])
Dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.number_input('Tenure (months)', min_value=0, step=1)
PhoneService = st.selectbox('Phone Service', ['No', 'Yes'])
MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes', 'Yes (Multiple)'])
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes'])
OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes'])
DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes'])
TechSupport = st.selectbox('Tech Support', ['No', 'Yes'])
StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes'])
StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('Paperless Billing', ['No', 'Yes'])
PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, step=0.1)
TotalCharges = st.number_input('Total Charges', min_value=0.0, step=0.1)

# Create dictionary of inputs
input_data = {
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
}

# OneHotEncoding
# Convert input dictionary to DataFrame
input_df = pd.DataFrame(input_data)
# Select categorical columns that need encoding (same as used in training)
categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']
# Apply OneHotEncoding to categorical columns
encoded_features = onehot_encoder.transform(input_df[categorical_columns])
# Convert the encoded data to DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out())
# Drop the original categorical columns from input data
input_df = input_df.drop(columns=categorical_columns)
# Merge input data with encoded features
final_input_df = pd.concat([input_df, encoded_df], axis=1)


# albel encodeing
# List of categorical features to be Label Encoded
label_encoded_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'PaperlessBilling', 'Churn'
]

for col in label_encoded_columns:
    if col in final_input_df.columns and col in label_encoders:
        # Convert column to string
        final_input_df[col] = final_input_df[col].astype(str)

        # Debug: Print unique values before encoding
        # print(f"Unique values in {col} before encoding: {final_input_df[col].unique()}")
        # print(f"Label Encoder Classes for {col}: {label_encoders[col].classes_}")

        # Ensure the encoder has seen the values
        known_classes = set(label_encoders[col].classes_)
        
        # Replace unseen values with a known class (e.g., most frequent one)
        final_input_df[col] = final_input_df[col].apply(lambda x: x if x in known_classes else list(known_classes)[0])

        # Apply transformation
        final_input_df[col] = label_encoders[col].transform(final_input_df[col])


# scaling the input data
input_scaled = scaler.transform(final_input_df)
input_scaled

# predict churn
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]

st.write("Probability of Churn:", prediction_probability)

if prediction_probability > 0.5:
    st.write("The customer is likely to churn")

else:
    st.write("The customer is not likely to churn")

