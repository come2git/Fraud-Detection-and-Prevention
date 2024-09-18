import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from prediction import predict
import joblib

# Load the saved OneHotEncoder and MinMaxScaler
encoder = joblib.load('onehot_encoding.pkl')


# Title of the app
st.title("Fraud Detection Prediction App")

# Input fields for user to fill in
amount = st.number_input('Transaction Amount', min_value=0.0, step=0.01)
initiating_branch = st.selectbox('Initiating Branch', ['Head Office Operations', 'Broad Street', 'Yaba', 'Head Office']) # Add other branches as needed
beneficiary_bank = st.selectbox('Beneficiary Bank', ['GTB', 'FCMB', 'UBA', 'OPay', 'First Bank', 'Access', 'Eco Bank', 'Mainstreet MFB', 'Zenith', 'Fidelity', 'Sterling', 'PolarisBank', 'Providus', 'PalmPay', 'MONIEPOINT MFB', 'Wema', 'Jubilee Life Mortgage', 'Stanbic IBTC', 'Kuda Bank', 'Keystone', 'Union', 'Suntrust', 'ALTERNATIVE BANK', 'Globus Bank', 'Ilora MFB', 'MoneyMaster PSB', 'Lotus Bank', 'FairMoney MFB', 'Paga', 'Unity', 'Jaiz', 'BANKLY MFB', 'VFD MFB', 'Heritage', 'Standard Chattered', 'DOT MICROFINANCE BANK', 'NEXIM BANK', 'CashConnect MFB', 'Alert MFB', 'MICHAEL OKPARA UNIAGRIC MFB', 'GoMoney', '9Payment Service Bank', 'Mint MFB', 'Coronation Merchant Bank', 'Titan Trust Bank', 'Richway MFB', 'Personal Trust MFB', 'Eyowo', 'Baines Credit', 'Empire Trust MFB', 'Spectrum MFB', 'Rand Bank', 'FinaTrust MFB', 'MoMo PSB']) 
status = st.selectbox('Transaction Status', ['Successful', 'Failed', 'Pending', 'Reversed', 'SuccesfulButFeeTaken', 'SuccesfulButFeeNotTaken', 'Dormant'])
transaction_type = st.selectbox('Transaction Type', ['Single', 'Bulk'])
response_description = st.selectbox('Response Description', ['Transaction Successful', 'Insufficient Funds', 'Exceeds Cash Limit', 'Error sending to Core Banking. The process cannot access the file isoClient.ser because it is being used by another process.', 'Exceeds Withdrawal Limit', 'Unable to locate record', 'Error', 'Unable to locate record', '0 Response From Switch', 'Timeout waiting for response from destination', 'Switch Unavailable', 'System malfunction', 'Do 0t ho0r', 'Invalid Transaction', 'Originator account number does not exist. Kindly reconfirm and try again', 'Transfer limit Exceeded', 'Main Transaction Was Successful But Fee Was 0t Taken. Reason is:', 'Awaiting confirmation status.', 'Exceeds withdrawal frequency', 'Beneficiary Bank 0t available', 'Format error', 'Approved or completed successfully', 'Account Name Mismatch', 'Invalid Account', 'Transaction not permitted to sender', '0 Response From Core Banking', 'Transaction violates account tier level restrictions', 'End Of Day Operation is running / HeadOffice branch is closed', 'Status Unknown, contact admin', 'Invalid Request', 'Fee Was Not Taken. Reason is: Invalid Amount', 'Originator account number does not exist. Kindly reconfirm and try again', 'Incomplete EChannel configuration. Kindly setup service on Bankone', 'Main Transaction Was Successful But Fee Was Not Taken. Reason is: No Check Account'])
gateway = st.selectbox('Gateway', ['NIBSS', 'EazyPay', 'ISW'])
hour = st.slider('Hour of Transaction', 0, 23, 0)
day = st.slider('Day of the Month', 1, 31, 1)
month = st.slider('Month of the Year', 1, 12, 1)

scaler = MinMaxScaler()
# When the user clicks the "Predict" button
if st.button("Predict Fraud"):

    # Create a dictionary with the user inputs
    user_data = {
        'Amount': [amount],
        'Initiating Branch': [initiating_branch],
        'Beneficiary Bank': [beneficiary_bank],
        'Status': [status],
        'Type': [transaction_type],
        'Response Description': [response_description],
        'Gateway': [gateway],
        'Hour': [hour],
        'Day': [day],
        'Month': [month]
    }

    # Convert to a DataFrame
    df = pd.DataFrame(user_data)

    # Apply OneHotEncoder to the categorical features
    categorical_columns = ['Initiating Branch', 'Beneficiary Bank', 'Status', 'Type', 'Response Description', 'Gateway']
    encoded_df = encoder.transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_df.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    # Normalize the numeric features (Amount, Hour, Day, Month)
    numeric_features = ['Amount', 'Hour', 'Day', 'Month']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Combine encoded and normalized features
    input_data = pd.concat([df[numeric_features], encoded_df], axis=1)

    # Make prediction using the predict function from prediction.py
    prediction = predict(input_data)

    # Display the result
    if prediction[0] == 0:
        st.success("This transaction is not fraudulent.")
    else:
        st.error("This transaction is fraudulent.")
