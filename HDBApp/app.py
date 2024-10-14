import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("""
    <style>
    /* Change the background color of the entire app */
    .stApp {
        background-color: #91caf9;  /* Light blue background */
    }

    /* Change font color globally */
    .stTextInput > div > input, .stNumberInput > div > input, .stSelectbox > div > div > div {
        color: #f0f8ff !important;  /* Steel blue font color for inputs */
        font-size: 16px;  /* Adjust font size */
    }
    
    /* Customize input boxes */
    input, select {
        background-color: #ffffff;  /* White background for inputs */
        border: 2px solid #4682b4;  /* Steel blue border for inputs */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px;  /* Add some padding inside the inputs */
    }

    /* Style for labels */
    .stMarkdown h1, h2, h3, h4, h5, h6, p {
        color: #1b3344;  /* Dark blue font color for labels */
    }

    /* Style for buttons */
    .stButton > button {
        background-color: #4682b4 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Function to predict price (placeholder)
def predict_price(street_name, block, town, flat_type, flat_model, floor_area, lease_commence_date, storey_range):
    # Placeholder for actual prediction logic
    return np.random.randint(100000, 500000)

# Streamlit UI
st.title("HDB Resale Price Prediction")

# User input fields
street_name = st.text_input("Street Name")
block = st.text_input("Block Number")

# Dropdown options for town, flat type, and model
towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
         'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
         'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
         'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
         'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
         'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
         'PUNGGOL']  
flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
               'Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace',
               '2-Room', 'Improved-Maisonette', 'Multi Generation',
               'Premium Apartment', 'Adjoined flat', 'Premium Maisonette',
               'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft',
               '3Gen']

town = st.selectbox("Town", towns)
flat_type = st.selectbox("Flat Type", flat_types)
flat_model = st.selectbox("Flat Model", flat_models)

floor_area = st.number_input("Floor Area (sqm)", min_value=0.0)
lease_commence_date = st.number_input("Lease Commence Date", min_value=1990, max_value=2023)
storey_range = st.text_input("Storey Range ('Value1' TO 'Value2')")

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(street_name, block, town, flat_type, flat_model, floor_area, lease_commence_date, storey_range)
    st.success(f"Predicted Resale Price: ${predicted_price:,.2f}")

    # Sample data for plotting (replace with your actual data)
    x = np.arange(10)  # X values from 0 to 9
    y = np.random.rand(10) * 100  # Random Y values scaled for better visibility
    
    # Displaying a simple graph
    plt.figure(figsize=(10, 4))
    plt.bar(x, y, color='skyblue')  # Change bar color
    plt.title("Sample Bar Graph", fontsize=16)
    plt.xlabel("Sample X Values", fontsize=12)
    plt.ylabel("Sample Y Values", fontsize=12)
    plt.xticks(x)  # Show ticks for better alignment
    plt.tight_layout()  # Fit layout nicely
    st.pyplot(plt)

