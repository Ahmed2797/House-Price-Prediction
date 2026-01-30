# House-Price-Prediction

## 🏠 House Price Prediction — One File ML Project

A **single-file machine learning project** for predicting house prices using  
**feature engineering + Random Forest + XGBoost**.

This project focuses on **correct prediction logic**, ensuring the model
can **reliably predict when a user provides raw input**.

---

## Dataset

    "https://drive.google.com/file/d/1iStp3wK4LoIWU958iLekNFMsax9ZSEKq/view?usp=sharing"

## 🎯 Goal

- Train a regression model on engineered features
- Predict house price from **raw user input**
- Use **only one Python file**
- Avoid train–test feature mismatch

---

## 🧠 Golden Rule (Very Important)

> ❗ Model only understands the features it was trained on  
> ❗ User input must be converted into those same features  

## Feature Enginering

    new_df = df.copy()
    new_df['sale_year'] = pd.to_datetime(new_df['date']).dt.year
    new_df['house_age'] = new_df['sale_year'] - new_df['yr_built']


    # new_df['effective_age'] = new_df.apply(lambda x: x['sale_year'] - max(x['yr_built'], x['yr_renovated']), axis=1)
    new_df['effective_age'] = np.where(
        new_df['yr_renovated'] > 0,
        new_df['sale_year'] - new_df['yr_renovated'],
        new_df['sale_year'] - new_df['yr_built']
    )

    new_df["condition_x_age"] = new_df["condition"] / (new_df["effective_age"] + 1)
    new_df["view_x_floor"] = new_df["view"] * new_df["floors"]
    new_df["luxury_score"] = (
        new_df["waterfront"]*5 +
        new_df["view"]*3 +
        new_df["condition"]*2
    )


    city_mean_price = new_df.groupby('city')['price'].mean()
    new_df['city_encoded'] = new_df['city'].map(city_mean_price)
    new_df['city_encoded'] = new_df['city_encoded'].round(2)


    ## Total rooms
    new_df['total_rooms'] = new_df['bedrooms'] + new_df['bathrooms']

    ## Average room size *********************
    new_df['avg_room_size'] = new_df['sqft_living'] / new_df['total_rooms']
    new_df['avg_room_size'] = new_df['avg_room_size'].round(2)


    ## statezip encode_mean******************************
    new_df['state_code'] = new_df['statezip'].str.extract(r'(\d+)')
    new_df['state_code'] = new_df['state_code'].astype('int')

    zip_price = new_df.groupby('state_code')['price'].mean().round(2)
    new_df['statezip_price_mean'] = new_df['state_code'].map(zip_price)

    ## Floors************************************
    # new_df['floors_x_view'] = new_df['floors'] * new_df['view']
    new_df["sqft_per_floor"] = new_df["sqft_living"] / new_df["floors"]

    new_df['price_log'] = np.log(new_df['price'])
    new_df['price_log'] = new_df['price_log'].round(2)
