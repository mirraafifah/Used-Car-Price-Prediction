import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load data
@st.cache
def load_data():
    return pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

data = load_data()

# Feature engineering
data['age'] = 2024 - data['year']
data['log_km_driven'] = np.log(data['km_driven'] + 1)
data['log_selling_price'] = np.log(data['selling_price'] + 1)
data['log_age'] = np.log(data['age'] + 1)

# Feature selection
selected_features = ['log_km_driven', 'log_age', 'fuel', 'seller_type', 'transmission', 'owner']
X = data[selected_features]
y = data['log_selling_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['log_km_driven', 'log_age']
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor())])

# Train the model
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit App
st.title('Used Car Price Prediction')

st.write('Mean Squared Error:', mse)

# Input features
input_features = {}
input_features['log_km_driven'] = st.number_input('Log Kilometers Driven', value=0.0)
input_features['log_age'] = st.number_input('Log Age of Car', value=0.0)

fuel_options = ['Petrol', 'Diesel', 'CNG', 'LPG']
input_features['fuel'] = st.selectbox('Fuel Type', fuel_options)

seller_type_options = ['Individual', 'Dealer', 'Trustmark Dealer']
input_features['seller_type'] = st.selectbox('Seller Type', seller_type_options)

transmission_options = ['Manual', 'Automatic']
input_features['transmission'] = st.selectbox('Transmission', transmission_options)

owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
input_features['owner'] = st.selectbox('Owner', owner_options)

# Predict
if st.button('Predict'):
    input_data = pd.DataFrame([input_features])
    prediction = np.exp(model.predict(input_data))[0] - 1
    st.success(f'Predicted Selling Price: Rs {prediction:.2f}')