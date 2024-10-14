import streamlit as st
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Load your trained models (excluding decision tree)
def load_models():
    try:
        meta_model = joblib.load('meta_model.sav')
        rf = joblib.load('random_forest_model.sav')
        gb = joblib.load('gradient_boosting_model.sav')

        # Load XGBoost model using its load_model method
        xgb = XGBRegressor()  # Create an instance of XGBRegressor
        xgb.load_model('xgb_model.json')  # Load the model from JSON file

        lr = joblib.load('linear_regression_model.sav')
        dt = joblib.load('decision_tree_model.sav')

        scaler = joblib.load('scaler.pkl')

        return rf, gb, xgb, lr, dt, meta_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None
    
rf, gb, xgb, lr, dt, meta_model, scaler = load_models()

# Check if models are loaded correctly
if rf is not None and gb is not None and xgb is not None and lr is not None and dt is not None and meta_model is not None:
    # Streamlit app title
    st.title("Resale Price Prediction using Meta-Model")

    # Load your validation DataFrame here
    # Replace 'your_validation_data.csv' with the path to your actual validation CSV file
    val_df = pd.read_csv('C:\\Users\\Krithika JK\\Documents\\GitHub\\FYP\\data\\hdb_val.csv')  # Ensure you have the correct path

    # Display the first few rows of the validation DataFrame
    st.header("Validation Data Preview")
    st.write(val_df.head(5))  # Display the first 5 rows of the validation DataFrame

    # Validate with existing validation DataFrame
    st.header("Evaluate on Validation DataFrame")

    if st.button("Evaluate"):
        # Prepare the validation DataFrame similarly to input_df
        val_input_df = val_df.copy()

        # One-hot encoding for categorical variables in validation set
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

        for town in towns:
            val_input_df[f'town_{town}'] = val_input_df['town'].apply(lambda x: 1 if x == town else 0)

        for flat_type in flat_types:
            val_input_df[f'flat_type_{flat_type}'] = val_input_df['flat_type'].apply(lambda x: 1 if x == flat_type else 0)

        for flat_model in flat_models:
            val_input_df[f'flat_model_{flat_model}'] = val_input_df['flat_model'].apply(lambda x: 1 if x == flat_model else 0)

        # Ensure the expected columns for the validation DataFrame
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

        val_input_df = val_input_df.reindex(columns=expected_columns, fill_value=0)

        st.write(val_input_df.head(5))

        val_input_scaled = scaler.transform(val_input_df.values)  # Scale validation data


        # Make predictions on the validation DataFrame
        rf_val_pred = rf.predict(val_input_scaled)
        gb_val_pred = gb.predict(val_input_scaled)
        xgb_val_pred = xgb.predict(val_input_scaled)
        lr_val_pred = lr.predict(val_input_scaled)
        dt_val_pred = dt.predict(val_input_scaled)

        # Combine predictions for meta-model
        X_val_meta = np.column_stack((rf_val_pred, gb_val_pred, lr_val_pred, xgb_val_pred, dt_val_pred))

        # Final predictions using meta-model
        y_val_pred = meta_model.predict(X_val_meta)

        # Calculate evaluation metrics
        rmse = root_mean_squared_error(val_df['resale_price'], y_val_pred)
        mae = mean_absolute_error(val_df['resale_price'], y_val_pred)
        r2 = r2_score(val_df['resale_price'], y_val_pred)

        # Display the evaluation metrics
        st.write("### Evaluation Metrics on Validation DataFrame:")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**R²:** {r2:.2f}")

else:
    st.error("Models could not be loaded.")
