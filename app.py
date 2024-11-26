import streamlit as st
import pickle
import numpy as np
import predict  

new_model = 'model.bin'
dv_file = 'dv.bin'

# Load the model, dv
with open(new_model, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in_dv:
    dv = pickle.load(f_in_dv)

# App title and description
st.title("Second-Hand LAPTOP Price Predition Using ML from Sri Lanka Market")
st.write("Provide the required inputs and get predictions from the model.")

dicts = {}

# Numerical Input
dicts['laptop'] = st.selectbox("Laptop Brand:", ["Apple","hp","Acer", "Asus","Dell","Lenovo","Toshiba"])
dicts['ram'] = st.number_input("RAM:", min_value=2.0, max_value=64.0, step=10.0)
dicts['cpu_company'] = st.selectbox("Processor:", ["intel", "amd"])
dicts['cpu_freq'] = st.number_input("CPU Frequency", min_value=0.9, max_value=3.6, step=30.0)
dicts['primarystoragetype'] = st.selectbox("Primary Storage Type", ["ssd", "flash_storage", "hdd", "hybrid"])
dicts['gpu_company'] = st.selectbox("GPU Brand", ["intel", "amd", "nvidia"])
dicts['category'] = st.selectbox("Laptop Category", ["portable", "standard", "heavy_duty", "other"])
dicts['os'] = st.selectbox("OS", ["windows", "mac", "other"])
dicts['resolution'] = st.selectbox("Resolution", ["quadhd", "hd", "fullhd", "ultrahd"])
dicts['gen'] = st.selectbox("Generation", ["i3", "i5", "i7", "amd"])
dicts['primarystorage'] = st.number_input("Storage Size", min_value=8.0, max_value=2048.0, step=10.0)

# Prediction
if st.button("Predict"):
    X = dv.transform([dicts])
    price = model.predict(X)
    final_price = np.expm1(price[0])    
    
    st.write(f"Price of this {dicts['laptop']} laptop is : {round(final_price, 3)} LKR")