import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

print("Joblib berhasil diimpor!")

# Load the saved model and encoders
model = joblib.load('toyota_model.pkl')
m_encoder = LabelEncoder()
t_encoder = LabelEncoder()
f_encoder = LabelEncoder()

# Predefine the unique values for encoding (based on your dataset)
model_labels = [' Corolla', ' Yaris', ' Aygo']  # Example models
transmission_labels = ['Manual', 'Automatic']
fuel_type_labels = ['Petrol', 'Diesel', 'Hybrid']

# Fit encoders with predefined labels
m_encoder.fit(model_labels)
t_encoder.fit(transmission_labels)
f_encoder.fit(fuel_type_labels)

# Streamlit app interface
st.title("Prediksi Harga Mobil Bekas")

# Input fields
st.header("Masukkan Detail Mobil")
model_input = st.selectbox("Model Mobil", model_labels)
year = st.number_input("Tahun Pembuatan", min_value=2000, max_value=2025, step=1)
transmission = st.selectbox("Transmisi", transmission_labels)
mileage = st.number_input("Jarak Tempuh (miles)", min_value=0)
fuel_type = st.selectbox("Tipe Bahan Bakar", fuel_type_labels)
tax = st.number_input("Pajak", min_value=0)
mpg = st.number_input("MPG (Miles per Gallon)", min_value=0.0, format="%.1f")
engine_size = st.number_input("Ukuran Mesin (Liter)", min_value=0.0, format="%.1f")

# Predict button
if st.button("Prediksi Harga"):
    # Prepare input data
    input_data = {
        'model': model_input,
        'year': year,
        'transmission': transmission,
        'mileage': mileage,
        'fuelType': fuel_type,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply label encoding
    input_df['model'] = m_encoder.transform(input_df['model'])
    input_df['transmission'] = t_encoder.transform(input_df['transmission'])
    input_df['fuelType'] = f_encoder.transform(input_df['fuelType'])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_formatted = round(prediction[0], 2)

    # Display result
    st.success(f"Prediksi Harga Mobil: Rp {prediction_formatted}")
