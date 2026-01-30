import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from helper import feature_engineering

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv('HousePrice.csv')
df = df[df['price'] > 0].copy()

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
df = feature_engineering(df)

# --------------------------------------------------
# Train-Test Split (IMPORTANT: before target encoding)
# --------------------------------------------------
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = X_train.copy()
train_data['price'] = y_train

test_data = X_test.copy()
test_data['price'] = y_test

# --------------------------------------------------
# Target Encoding (CITY)
# --------------------------------------------------
city_mean_price = train_data.groupby('city')['price'].mean().round(2)
GLOBAL_CITY_MEAN = city_mean_price.mean()
print(f"GLOBAL_CITY_MEAN {GLOBAL_CITY_MEAN}")

train_data['city_encoded'] = train_data['city'].map(city_mean_price)
test_data['city_encoded'] = test_data['city'].map(city_mean_price).fillna(GLOBAL_CITY_MEAN)

# --------------------------------------------------
# Target Encoding (ZIP / state_code)
# --------------------------------------------------
zip_price_mean = train_data.groupby('state_code')['price'].mean().round(2)
GLOBAL_ZIP_MEAN = zip_price_mean.mean()
print(f"GLOBAL_ZIP_MEAN {GLOBAL_ZIP_MEAN}")

train_data['statezip_price_mean'] = train_data['state_code'].map(zip_price_mean)
test_data['statezip_price_mean'] = test_data['state_code'].map(zip_price_mean).fillna(GLOBAL_ZIP_MEAN)

# --------------------------------------------------
# Log Transform Target
# --------------------------------------------------
train_data['price_log'] = np.log(train_data['price']).round(4)
test_data['price_log'] = np.log(test_data['price']).round(4)

# --------------------------------------------------
# Drop Unused Columns
# --------------------------------------------------
DROP_COLS = [
    'sqft_lot', 'floors', 'sale_year', 'state_code',
    'price', 'date', 'city', 'statezip', 'country',
    'yr_built', 'yr_renovated', 'street', 'sqft_living'
]

train_data.drop(columns=DROP_COLS, inplace=True)
test_data.drop(columns=DROP_COLS, inplace=True)

# --------------------------------------------------
# Clean Data
# --------------------------------------------------
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# --------------------------------------------------
# Final Train/Test
# --------------------------------------------------
X_train = train_data.drop(columns=['price_log'])
y_train = train_data['price_log']

X_test = test_data.drop(columns=['price_log'])
y_test = test_data['price_log']


# --------------------------------------------------
# XGBoost Model
# --------------------------------------------------
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    objective='reg:squarederror',
    eval_metric='rmse',
    early_stopping_rounds=50,
    random_state=42
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_pred = xgb.predict(X_test)

print("XGBoost R2:", r2_score(y_test, xgb_pred))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))
print("Best Iteration:", xgb.best_iteration)

# --------------------------------------------------
# Random Forest Model
# --------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

# --------------------------------------------------
# Save Everything (VERY IMPORTANT)
# --------------------------------------------------
FEATURES = X_train.columns.tolist()

joblib.dump(rf, 'rf_model.pkl')
joblib.dump(xgb, 'xgb_model.pkl')

joblib.dump(city_mean_price, 'city_mean_price.pkl')
joblib.dump(GLOBAL_CITY_MEAN, 'global_city_mean.pkl')

joblib.dump(zip_price_mean, 'zip_price_mean.pkl')
joblib.dump(GLOBAL_ZIP_MEAN, 'global_zip_mean.pkl')

joblib.dump(FEATURES, 'features.pkl')

print("All models & encoders saved successfully")
