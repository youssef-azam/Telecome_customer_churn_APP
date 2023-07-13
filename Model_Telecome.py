import streamlit as st
import pandas as pd
import pickle

# Load the data into a DataFrame (replace 'df' with your actual DataFrame)
df = pd.read_csv(r"https://raw.githubusercontent.com/youssef-azam/Telecome_customer_churn_APP/main/telecom_customer_churn.csv")

# Mapping dictionary for categorical features
mapping = {
    'Married': {'Yes': 1, 'No': 0},
    'Offer': {'None': 0, 'Offer B': 2, 'Offer E': 5, 'Offer D': 4, 'Offer A': 1, 'Offer C': 3},
    'Internet Service': {'Yes': 1},
    'Online Security': {'No': 0, 'Yes': 1},
    'Online Backup': {'No': 0, 'Yes': 1},
    'Device Protection Plan': {'No': 0, 'Yes': 1},
    'Premium Tech Support': {'No': 0, 'Yes': 1},
    'Contract': {'Month-to-Month': 0, 'Two Year': 2, 'One Year': 1},
    'Paperless Billing': {'Yes': 1, 'No': 0}
}

# Apply mapping to convert categorical features to numeric values
df.replace(mapping, inplace=True)

# Choose the best features
X = df[['Married', 'Number of Dependents', 'Number of Referrals',
        'Tenure in Months', 'Offer', 'Internet Service', 'Online Security',
        'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
        'Contract', 'Paperless Billing', 'Total Charges',
        'Total Long Distance Charges', 'Total Revenue', 'Age']]
y = df['Customer Status']

# Load the pickled model
with open(r"RF.pkl", 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit app
st.set_page_config(layout="wide")  # Set wide layout for better streamlit experience
st.markdown(
    "<h1 style='text-align: center;'>Customer Churn Prediction📈</h1>",
    unsafe_allow_html=True
)
st.write("This app predicts customer churn based on selected features.")



# Use columns for better organization
col1, col2 = st.columns(2)

# Column 1
with col1:
    married = st.selectbox("Married", ['Yes', 'No'])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    referrals = st.number_input("Number of Referrals", min_value=0, max_value=10, value=0)
    tenure = st.number_input("Tenure in Months", min_value=0, max_value=100, value=0)
    offer = st.selectbox("Offer", ['None', 'Offer B', 'Offer E', 'Offer D', 'Offer A', 'Offer C'])
    internet_service = st.selectbox("Internet Service", ['Yes', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No'])

# Column 2
with col2:
    device_protection = st.selectbox("Device Protection Plan", ['Yes', 'No'])
    tech_support = st.selectbox("Premium Tech Support", ['Yes', 'No'])
    contract = st.selectbox("Contract", ['Month-to-Month', 'Two Year', 'One Year'])
    billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=0.0)
    long_distance_charges = st.number_input("Total Long Distance Charges", min_value=0.0, max_value=1000.0, value=0.0)
    total_revenue = st.number_input("Total Revenue", min_value=0.0, max_value=10000.0, value=0.0)
    age = st.number_input("Age", min_value=0, max_value=100, value=0)

# Make predictions
input_data = [[married, dependents, referrals, tenure, offer, internet_service, online_security,
               online_backup, device_protection, tech_support, contract, billing, total_charges,
               long_distance_charges, total_revenue, age]]

# Apply mapping to convert categorical features to numeric values
input_data_mapped = []
for i in range(len(input_data[0])):
    col_name = X.columns[i]
    col_mapping = mapping.get(col_name)
    if col_mapping is not None:
        input_data_mapped.append(col_mapping.get(input_data[0][i]))
    else:
        input_data_mapped.append(input_data[0][i])

prediction = model.predict([input_data_mapped])

if st.button("Predict"):
    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    if churn_status == "Churn":
        st.error(f"The predicted churn status for the customer is: {churn_status}")
    else:
        st.success(f"The predicted churn status for the customer is: {churn_status}")

st.text("Dashboard Using Tableau")
# Display an image of yourself
st.image("Dashboard 2.png", width=200)
# Display an image of yourself
st.image("Dashboard 1 (1).png", width=200)

st.text("Create By:- Youssef Azam ")
# Display a text description of yourself
st.text("My name is Youssef Azam, I'm currently studying Computer Science and Artificial Intelligence
. I'm passionate about the field of Data Science.")

# Display a link to your LinkedIn profile
st.text("LinkedIn Profile: ")
st.markdown("[Youssef Azam](https://www.linkedin.com/in/youssef-azam-a36816231)")

