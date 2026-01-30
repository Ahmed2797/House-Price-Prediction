import joblib
import numpy as np
import pandas as pd
from helper import feature_engineering
# --------------------------------------------------
# Load models & encoders
# --------------------------------------------------
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

FEATURES = joblib.load('features.pkl')

city_mean_price = joblib.load('city_mean_price.pkl')
GLOBAL_CITY_MEAN = joblib.load('global_city_mean.pkl')

zip_mean_price = joblib.load('zip_price_mean.pkl')
GLOBAL_ZIP_MEAN = joblib.load('global_zip_mean.pkl')


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_price(user_input: dict):
    """
    user_input: dictionary with raw house features
    returns: predicted house price (original scale)
    """

    df = pd.DataFrame([user_input])

    # --------------------------------------------------
    # Feature Engineering (same as training)
    # --------------------------------------------------
    df = feature_engineering(df)

    # --------------------------------------------------
    # Target Encoding (SAFE)
    # --------------------------------------------------
    df['city_encoded'] = (
        df['city']
        .map(city_mean_price)
        .fillna(GLOBAL_CITY_MEAN)
    )

    df['statezip_price_mean'] = (
        df['state_code']
        .map(zip_mean_price)
        .fillna(GLOBAL_ZIP_MEAN)
    )

    drop_cols = [
    'sqft_lot','floors','sale_year',
    'date','city','statezip','country',
    'yr_built','yr_renovated','state_code','sqft_living'
    ]
    df = df.drop(columns=drop_cols)


    # --------------------------------------------------
    # Select features in correct order
    # --------------------------------------------------
    X = df # df[FEATURES]

    # --------------------------------------------------
    # Model predictions (log-scale)
    # --------------------------------------------------
    rf_pred_log = rf_model.predict(X)
    xgb_pred_log = xgb_model.predict(X)

    final_log_price = (rf_pred_log + xgb_pred_log) / 2

    # --------------------------------------------------
    # Convert back to original price
    # --------------------------------------------------
    rf_pred_log = np.exp(rf_pred_log)
    print(f"rf_pred_log{rf_pred_log}")

    xgb_pred_log = np.exp(xgb_pred_log)
    print(f"xgb_pred_log{xgb_pred_log}")

    final_price = np.exp(final_log_price)
    print(final_price)

    return {
        "predicted_price": round(float(final_price), 2)
    }


user_input = {
    'date': '2014-05-02',
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft_living': 1800,
    'sqft_lot': 5000,
    'floors': 2,
    'waterfront': 0,
    'view': 1,
    'condition': 4,
    'sqft_above': 1800,
    'sqft_basement': 0,
    'yr_built': 2005,
    'yr_renovated': 0,
    'city': 'Seattle',
    'statezip': 'WA 98103',
    'country': 'USA'
}

# final_price = predict_price(user_input)
# print(final_price)