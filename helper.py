import numpy as np
import pandas  as pd


def feature_engineering(df:pd.DataFrame=None)->pd.DataFrame:
    df = df.copy()

    # date
    df['sale_year'] = pd.to_datetime(df['date']).dt.year
    # age
    df['house_age'] = df['sale_year'] - df['yr_built']

    df['effective_age'] = np.where(
        df['yr_renovated'] > 0,
        df['sale_year'] - df['yr_renovated'],
        df['sale_year'] - df['yr_built']
    )

    df['avg_room_size'] = df['sqft_living'] / (df['bedrooms'] + df['bathrooms'])
    df['avg_room_size'] = df['avg_room_size'].round(2)

    df['sqft_per_floor'] = df['sqft_living'] / df['floors']
    df['sqft_per_floor'] = df['sqft_per_floor'].round(2)


    df['state_code'] = df['statezip'].str.extract(r'(\d+)').astype(int)

    return df
