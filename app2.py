import streamlit as st
import numpy as np
import json
import requests
from geopy.distance import geodesic
import pickle
import pandas as pd
import statistics

# Load pre-trained model from pickle file
with open('model_rf.pkl', 'rb') as file:
    rf_model = pickle.load(file)  # Load your Random Forest model

# Load the scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load MRT, Supermarket, School, and Hawker Center location data
mrt_location = pd.read_csv('data/mrt_address.csv')
supermarket_location = pd.read_csv('data/shops_address.csv')
school_location = pd.read_csv('data/schools_address.csv')
hawker_location = pd.read_csv('data/hawkers_address.csv')

# Function to retrieve latitude and longitude for an address
def get_lat_long(address):
    try:
        url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
        response = requests.get(url)
        data = response.json()
        if data['found'] > 0:
            latitude = data['results'][0]['LATITUDE']
            longitude = data['results'][0]['LONGITUDE']
            return latitude, longitude
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {address}: {e}")
        return None, None

# Function to calculate the median of storey range
def get_median(x):
    split_list = x.split(' TO ')
    float_list = [float(i) for i in split_list]
    median = statistics.median(float_list)
    return median

def predict_price(street_name, block, floor_area_sqm, lease_commence_date, storey_range, year_of_sale, flat_model, town):
    # Calculate lease_remain_years using lease_commence_date
    calculated_remaining_lease = 99 - (year_of_sale - lease_commence_date)

    # Year of sale - current year
    year_of_sale = 2024

    # Calculate median of storey_range
    storey_median = get_median(storey_range)

    # Get the address by joining the block number and the street name
    address = f"{block} {street_name}"
    latitude, longitude = get_lat_long(address)

    if latitude is None or longitude is None:
        st.error("Could not retrieve coordinates for the provided address.")
        return None

    origin = (latitude, longitude)

    # Calculate distances to amenities
    mrt_lat = mrt_location['latitude']
    mrt_long = mrt_location['longitude']
    list_of_mrt_coordinates = [(lat, long) for lat, long in zip(mrt_lat, mrt_long)]
    nearest_mrt_distance = min([geodesic(origin, mrt).meters for mrt in list_of_mrt_coordinates])

    supermarket_lat = supermarket_location['latitude']
    supermarket_long = supermarket_location['longitude']
    list_of_supermarket_coordinates = [(lat, long) for lat, long in zip(supermarket_lat, supermarket_long)]
    nearest_supermarket_distance = min([geodesic(origin, market).meters for market in list_of_supermarket_coordinates])

    school_lat = school_location['latitude']
    school_long = school_location['longitude']
    list_of_school_coordinates = [(lat, long) for lat, long in zip(school_lat, school_long)]
    nearest_school_distance = min([geodesic(origin, school).meters for school in list_of_school_coordinates])

    hawker_lat = hawker_location['latitude']
    hawker_long = hawker_location['longitude']
    list_of_hawker_coordinates = [(lat, long) for lat, long in zip(hawker_lat, hawker_long)]
    nearest_hawkers_distance = min([geodesic(origin, hawker).meters for hawker in list_of_hawker_coordinates])

    # Calculate distance from CBD
    cbd_distance = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

    # Prepare input data for the model
    new_sample = np.array([[cbd_distance, nearest_mrt_distance, floor_area_sqm, calculated_remaining_lease, year_of_sale,
                            storey_median, nearest_supermarket_distance, nearest_school_distance, nearest_hawkers_distance]])

    # One-hot encode flat_type, flat_model, and town
    flat_info = pd.DataFrame({
        'flat_type': [flat_type],
        'flat_model': [flat_model],
        'town': [town]
    })

    flat_info_encoded = pd.get_dummies(flat_info, drop_first=True)

    # Combine with the new sample data
    new_sample = np.concatenate((new_sample, flat_info_encoded), axis=1)

    # Scale the input data
    new_sample_scaled = scaler.transform(new_sample)

    # Predict resale price using the loaded model
    new_pred = rf_model.predict(new_sample_scaled)[0]
    return new_pred  # Direct output since no log transformation was used

# Streamlit App
st.title("HDB Resale Price Prediction")

# Hardcoded preset values for testing
street_name = st.text_input("Street Name:", value="ANG MO KIO AVE 1")
block = st.text_input("Block Number:", value="309")
floor_area_sqm = st.number_input("Floor Area (sqm):", value=31.0, min_value=0.0)
lease_commence_date = st.number_input("Lease Commence Date:", value=1977, min_value=0, max_value=2023)
storey_range = st.text_input("Storey Range ('Value1' TO 'Value2'):", value="10 TO 12")
year_of_sale = st.number_input("Year of Sale:", value=2024, min_value=2000, max_value=2024)
flat_type = st.selectbox("Flat Type:", options=["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"], index=1)
flat_model = st.selectbox("Flat Model:", options=["Model A", "Model B", "Model C"], index=0)
town = st.selectbox("Town:", options=["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH"], index=0)

if st.button("Predict Price"):
    predicted_price = predict_price(
        street_name,
        block,
        floor_area_sqm,
        lease_commence_date,
        storey_range,
        year_of_sale,
        flat_type,
        flat_model,
        town,
    )
    if predicted_price is not None:
        st.success(f"Predicted Resale Price: ${predicted_price:,.2f}")
