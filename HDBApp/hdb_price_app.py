import streamlit as st
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBRegressor
import requests
from geopy.distance import geodesic
import pydeck as pdk  
from PIL import Image

st.set_page_config(layout="wide")

# Town Coordinates
town_coordinates = {
    'ANG MO KIO': (1.375, 103.848), 
    'BEDOK': (1.324, 103.929), 
    'BISHAN': (1.358, 103.848), 
    'BUKIT BATOK': (1.366, 103.763),
    'BUKIT MERAH': (1.274, 103.822),
    'BUKIT TIMAH': (1.333, 103.776), 
    'CENTRAL AREA': (1.299, 103.849), 
    'CHOA CHU KANG': (1.389, 103.749), 
    'CLEMENTI': (1.317, 103.764),    
    'GEYLANG': (1.313, 103.867), 
    'HOUGANG': (1.367, 103.891), 
    'JURONG EAST': (1.334, 103.733), 
    'JURONG WEST': (1.350, 103.706),
    'KALLANG/WHAMPOA': (1.316, 103.857), 
    'MARINE PARADE': (1.304, 103.901), 
    'QUEENSTOWN': (1.290, 103.798), 
    'SENGKANG': (1.375, 103.895),
    'SERANGOON': (1.358, 103.870), 
    'TAMPINES': (1.351, 103.940), 
    'TOA PAYOH': (1.334, 103.846), 
    'WOODLANDS': (1.438, 103.786), 
    'YISHUN': (1.426, 103.836),
    'LIM CHU KANG': (1.437, 103.709), 
    'SEMBAWANG': (1.450, 103.834), 
    'BUKIT PANJANG': (1.377, 103.769), 
    'PASIR RIS': (1.372, 103.948),
    'PUNGGOL': (1.403, 103.903)
}

# Dropdown options for town, flat type, and flat model
towns = list(town_coordinates.keys())
flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
               'Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace',
               '2-Room', 'Improved-Maisonette', 'Multi Generation',
               'Premium Apartment', 'Adjoined flat', 'Premium Maisonette',
               'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft',
               '3Gen']

st.markdown("""<style>
    /* Change the background color of the entire app to a more pronounced gradient */
    .stApp {
        background: linear-gradient(to right, #ade8f4, #48cae4, #00b4d8, #0096c7);  /* Multiple shades of blue */
    }
    /* Style for labels */
    .stMarkdown h1, h2, h3, h4, h5, h6, p {
        color: #1b3344;  /* Dark blue font color for labels */
    }
    /* Style for buttons */
    .stButton > button {
        background-color: #e3f2fd 
        color: #1b3344 
        border: none;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
    }
</style>""", unsafe_allow_html=True)

# Load your trained models (excluding decision tree)
def load_models():
    try:
        meta_model = joblib.load('meta_model.pkl')
        rf = joblib.load('random_forest_model.pkl')
        gb = joblib.load('gradient_boosting_model.pkl')

        # Load XGBoost model using its load_model method
        xgb = XGBRegressor()  # Create an instance of XGBRegressor
        xgb.load_model('xgb_model.json')  # Load the model from JSON file

        lr = joblib.load('linear_regression_model.pkl')
        dt = joblib.load('decision_tree_model.pkl')
        scaler = joblib.load('scaler.pkl')

        return rf, gb, xgb, lr, dt, meta_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

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

def cal_lease_remaining_years(lease_commence_date, year_of_sale):
    return 99 - (year_of_sale - lease_commence_date)

def get_address(block, street_name):
    return f"{block} {street_name}"

rf, gb, xgb, lr, dt, meta_model, scaler = load_models()  # Make sure to also include xgb in the unpacking

# Check if models are loaded correctly
if rf is not None and gb is not None and xgb is not None and lr is not None and dt is not None and meta_model is not None:
    # Streamlit app title
    st.markdown("<h1 style='text-align: center; '>🏠 HDB Resale Price Prediction</h1>", unsafe_allow_html=True)

    st.header("Input Features")

    st.markdown("<h5>Enter the features of the HDB flat to estimate it's resale price</h5>", unsafe_allow_html=True)

    # Create two columns
    col1, col2, col3 = st.columns([1,1,2])

    # Using dropdowns for categorical inputs in the first column
    with col1:
        selected_town = st.selectbox("**Town**", towns, index=towns.index("JURONG WEST"))
        selected_flat_type = st.selectbox("**Flat Type**", flat_types, index=flat_types.index("5 ROOM"))
        selected_flat_model = st.selectbox("**Flat Model**", flat_models, index=flat_models.index("Improved"))

        block = st.text_input("**HDB Block**", value="909")
        street_name = st.text_input("**Street Name**", value="JURONG WEST ST 91")


    with col2:
        year_of_sale = st.number_input("**Year of Sale**", min_value=2000, max_value=2024, step=1, value=2019)
        storey_median = st.number_input("**Storey Median**", min_value=1, step=1, value=8)
        lease_commence_date = st.number_input("**Lease Commence Date**", min_value=1950, max_value=2023, step=1, value=1989)
        floor_area_sqm = st.number_input("**Floor Area (sqm)**", min_value=0.0, step=1.0, value=122.00)


    # Construct address
    address = get_address(block, street_name)

    latitude, longitude = get_lat_long(address)

    if latitude is None or longitude is None:
        st.error("Could not retrieve coordinates for the provided address.")
    else:
        origin = (latitude, longitude)

    # Load the location data for amenities
    mrt_location = pd.read_csv('C:\\Users\\Krithika JK\\Documents\\GitHub\\FYP\\data\\mrt_address.csv')  
    school_location = pd.read_csv('C:\\Users\\Krithika JK\\Documents\\GitHub\\FYP\\data\\schools_address.csv')  
    supermarket_location = pd.read_csv('C:\\Users\\Krithika JK\\Documents\\GitHub\\FYP\\data\\shops_address.csv')  
    hawker_location = pd.read_csv('C:\\Users\\Krithika JK\\Documents\\GitHub\\FYP\\data\\hawkers_address.csv')  

    # Calculate distances to amenities
    nearest_mrt_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in mrt_location.iterrows()])
    nearest_supermarket_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in supermarket_location.iterrows()])
    nearest_school_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in school_location.iterrows()])
    nearest_hawkers_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in hawker_location.iterrows()])

    # Calculate distance from CBD
    cbd_distance = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

    # Calculate lease remaining years using lease_commence_date
    calculated_remaining_lease = cal_lease_remaining_years(lease_commence_date, year_of_sale)


    # Create a DataFrame for the input data
    input_data = {
        'nearest_hawkers_distance': [nearest_hawkers_distance],
        'floor_area_sqm': [floor_area_sqm],
        'nearest_mrt_distance': [nearest_mrt_distance],
        'nearest_school_distance': [nearest_school_distance],
        'cbd_distance': [cbd_distance],
        'year_of_sale': [year_of_sale],
        'storey_median': [storey_median],
        'calculated_remaining_lease': [calculated_remaining_lease],
        'nearest_supermarket_distance': [nearest_supermarket_distance]
    }
    
    input_df = pd.DataFrame(input_data)

    for town in towns:
        input_df[f'town_{town}'] = 1 if selected_town == town else 0

    for flat_type in flat_types:
        input_df[f'flat_type_{flat_type}'] = 1 if selected_flat_type == flat_type else 0

    for flat_model in flat_models:
        input_df[f'flat_model_{flat_model}'] = 1 if selected_flat_model == flat_model else 0


    # One-hot encode categorical features
    # input_df_encoded = pd.get_dummies(input_df, columns=['town', 'flat_type', 'flat_model'], drop_first=True)

    # Define the expected columns based on your training data
    expected_columns = ['floor_area_sqm', 'nearest_supermarket_distance', 'nearest_school_distance', 'nearest_mrt_distance', 
                        'nearest_hawkers_distance', 'cbd_distance', 'year_of_sale', 'calculated_remaining_lease', 
                        'storey_median', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 
                        'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 
                        'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 
                        'town_KALLANG/WHAMPOA', 'town_LIM CHU KANG', 'town_MARINE PARADE', 'town_PASIR RIS', 
                        'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 
                        'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN', 'flat_model_3Gen', 
                        'flat_model_Adjoined flat', 'flat_model_Apartment', 'flat_model_DBSS', 'flat_model_Improved', 
                        'flat_model_Improved-Maisonette', 'flat_model_Maisonette', 'flat_model_Model A', 
                        'flat_model_Model A-Maisonette', 'flat_model_Model A2', 'flat_model_Multi Generation', 
                        'flat_model_New Generation', 'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft', 
                        'flat_model_Premium Maisonette', 'flat_model_Simplified', 'flat_model_Standard', 'flat_model_Terrace', 
                        'flat_model_Type S1', 'flat_model_Type S2', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM', 
                        'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI GENERATION']



    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    # st.write(input_df)

    input_scaled = scaler.transform(input_df)  # Scale validation data
    # st.write(input_scaled)

with col1:
    if st.button("Predict"):
        try:
            # Make predictions using the base models (excluding decision tree)
            rf_pred_new = rf.predict(input_scaled)
            gb_pred_new = gb.predict(input_scaled)
            xgb_pred_new = xgb.predict(input_scaled)
            lr_pred_new = lr.predict(input_scaled)
            dt_pred_new = dt.predict(input_scaled)

            # Combine the predictions into a single feature matrix for the meta-model
            X_new_meta = np.column_stack((rf_pred_new, gb_pred_new, lr_pred_new, dt_pred_new, xgb_pred_new))

            # Make the final prediction using the meta-model
            y_new_pred = meta_model.predict(X_new_meta)



            st.write("#### Predicted Resale Price: ${:.2f}".format(y_new_pred[0]))           # Display the predicted resale price


            # Prepare data for Pydeck chart
            town_df_base = input_df.copy().drop(columns=[                 # Create a copy of input_df 
                'nearest_supermarket_distance',
                'nearest_mrt_distance',
                'nearest_school_distance',
                'nearest_hawkers_distance'
            ])

            # Create a DataFrame for all towns
            town_predictions = []
            for town in towns:
                # Copy the base DataFrame
                town_df = town_df_base.copy()
                
                for current_town in towns:
                    town_df[f'town_{current_town}'] = 1 if current_town == town else 0

                # Get latitude and longitude for the current town from town_coordinates
                if town in town_coordinates:
                    latitude, longitude = town_coordinates[town]
                else:
                    st.error(f"Could not find coordinates for town: {town}.")
                    continue            

                if latitude is None or longitude is None:
                    st.error(f"Could not retrieve coordinates for town: {town}.")
                    continue
                
                # Calculate distances to amenities
                origin = (latitude, longitude)
                nearest_mrt_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in mrt_location.iterrows()])
                nearest_supermarket_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in supermarket_location.iterrows()])
                nearest_school_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in school_location.iterrows()])
                nearest_hawkers_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in hawker_location.iterrows()])

                # Calculate distance from CBD
                cbd_distance = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # Add the distances to the town_df
                town_df['nearest_mrt_distance'] = nearest_mrt_distance
                town_df['nearest_supermarket_distance'] = nearest_supermarket_distance
                town_df['nearest_school_distance'] = nearest_school_distance
                town_df['nearest_hawkers_distance'] = nearest_hawkers_distance


                town_df = town_df.reindex(columns=expected_columns, fill_value=0)
                # st.write(town_df)

                # Scale the input town_df
                town_scaled = scaler.transform(town_df)

                # Make predictions for the town
                town_rf_pred = rf.predict(town_scaled)
                town_gb_pred = gb.predict(town_scaled)
                town_xgb_pred = xgb.predict(town_scaled)
                town_lr_pred = lr.predict(town_scaled)
                town_dt_pred = dt.predict(town_scaled)

                # Combine predictions for meta-model
                town_X_meta = np.column_stack((town_rf_pred, town_gb_pred, town_lr_pred, town_dt_pred, town_xgb_pred))
                town_y_pred = meta_model.predict(town_X_meta)

                # Store predictions along with town coordinates
                town_predictions.append({
                    'town': town,
                    'latitude': latitude,
                    'longitude': longitude,
                    'predicted_price': town_y_pred[0],
                    'formatted_price': "${:.2f}".format(town_y_pred[0])
                })

            # Convert to DataFrame
            predictions_df = pd.DataFrame(town_predictions)
            predictions_df['normalized_price'] = (predictions_df['predicted_price'] - predictions_df['predicted_price'].min()) / (predictions_df['predicted_price'].max() - predictions_df['predicted_price'].min())

            # Function to map normalized prices to a color gradient from bright yellow to dark green
            def price_to_color(normalized_price):
                # Bright yellow color (255, 255, 0)
                start_color = [255, 255, 0]
                
                # Dark green color (0, 128, 0)
                end_color = [0, 64, 0]

                # Interpolate between bright yellow and dark green
                r = int(start_color[0] + (end_color[0] - start_color[0]) * normalized_price)
                g = int(start_color[1] + (end_color[1] - start_color[1]) * normalized_price)
                b = int(start_color[2] + (end_color[2] - start_color[2]) * normalized_price)

                return [r, g, b]

        
            predictions_df['color'] = predictions_df['normalized_price'].apply(price_to_color)

            # Create a Pydeck chart for all towns
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=1.3521,  # Singapore Latitude
                    longitude=103.8198,  # Singapore Longitude
                    zoom=10.5,
                    pitch=0,
                    map_style="mapbox://styles/mapbox/light-v11"  # Light mode map style

                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=predictions_df,
                        get_position="[longitude, latitude]",  # Ensure this matches your DataFrame structure
                        get_radius=700,
                        get_color="color",  # Use the 'color' column for the heatmap effect
                        pickable=True,
                        opacity=0.6,

                    )
                ],
                tooltip={
                    "html": "<b>Town:</b>{town}<br/><b>Predicted Price:</b> {formatted_price}",
                    "style": {"color": "white"},  # Adjust color as needed for visibility
                },
            )

            with col3:
                st.markdown("<h3 style='text-align: center;'>🔍 Predicted Resale Prices by Town</h3>", unsafe_allow_html=True)
                st.pydeck_chart(deck)

                # Generate a gradient image
                def create_gradient_image(width=400, height=20):
                    gradient = np.zeros((height, width, 3), dtype=np.uint8)
                    for x in range(width):
                        normalized_price = x / width  # Normalized price for color mapping
                        color = price_to_color(normalized_price)
                        gradient[:, x] = color
                    return Image.fromarray(gradient)

                # Create and display the gradient image
                gradient_image = create_gradient_image()
                st.image(gradient_image, use_column_width=True)

                # Add labels for the gradient bar with improved alignment
                st.markdown("""
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='color: 1b3344'>Low Price</span>
                                <span style='color: 1b3344'>High Price</span>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as prediction_error:
            st.error(f"Error making predictions: {prediction_error}")
