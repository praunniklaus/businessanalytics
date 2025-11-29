import numpy as np
import pandas as pd
import json
import re

# Distance Function to calculate distance between two points on Earth
# Used to calculate distance to landmarks
def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371  # Earth radius in kilometers
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

    
# Preprocessing Function
def preprocess():

    # Load Raw Data
    data_raw = pd.read_csv('data/listings.csv')

    # Columns to Keep
    keep_cols = [
    'host_since', 'host_total_listings_count', 'price', 'accommodates', 'bedrooms', 'beds', 'room_type', 
    'neighbourhood_group_cleansed', 'latitude', 'longitude', 'amenities', 'bathrooms_text', 
    'review_scores_rating', 'property_type','number_of_reviews',
    'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location', 'review_scores_value']

    # Select Columns (drop rest)
    data = data_raw[keep_cols].copy()


    # Convert host_since to host_experience
    reference_date = pd.to_datetime('2025-09-24')
    data['host_experience'] = (reference_date - pd.to_datetime(data['host_since'])).dt.days
    data = data.drop(columns=['host_since'])
    
    # Create professional host feature
    data['is_professional_host'] = (data['host_total_listings_count'] > 2).astype(int)
    data = data.drop(columns=['host_total_listings_count'])

    # Convert price to numeric
    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Drop entries without price
    data = data.dropna(subset=['price'])

    # Convert room_type to binary columns
    room_types = data['room_type'].unique()
    for room_type in room_types:
        if pd.notna(room_type):
            col_name = 'is_' + room_type.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('ö', 'oe').replace('ä', 'ae').replace('ü', 'ue')
            data[col_name] = (data['room_type'] == room_type).astype(int)
    data = data.drop(columns=['room_type'])


    # Convert neighbourhood_group_cleansed to binary columns
    neighbourhoods = data['neighbourhood_group_cleansed'].unique()
    for neighbourhood in neighbourhoods:
        if pd.notna(neighbourhood):
            col_name = 'is_' + neighbourhood.lower().replace(' ', '_').replace('-', '_').replace('ö', 'oe').replace('ä', 'ae').replace('ü', 'ue').replace('.', '')
            data[col_name] = (data['neighbourhood_group_cleansed'] == neighbourhood).astype(int)
    data = data.drop(columns=['neighbourhood_group_cleansed'])

    # Coordinates of landmarks
    landmarks = {
        'Brandenburger Tor': (52.516275, 13.377704),
        'Alexanderplatz': (52.521918, 13.413215),
        'Reichstagsgebäude': (52.518620, 13.376187),
        'Potsdamer Platz': (52.509168, 13.376641),
        'Siegessäule': (52.514543, 13.350119),
        'Tiergarten': (52.509778, 13.357260),
    }

    # Calculate distance to landmarks (in km)
    for landmark, (lat, lon) in landmarks.items():
        landmark_clean = landmark.lower().replace(' ', '_').replace('-', '_').replace('ö', 'oe').replace('ä', 'ae').replace('ü', 'ue')
        data[f'dist_{landmark_clean}'] = haversine_distance(data['latitude'], data['longitude'], lat, lon)
    data = data.drop(columns=['latitude', 'longitude'])
    
    # Convert amenities to binary columns (only top important amenities)
    important_amenities = [
        'Wifi', 'Kitchen', 'Air conditioning', 'Free parking', 'Washer', 'Dryer',
        'Heating', 'TV', 'Dishwasher', 'Refrigerator', 'Microwave', 'Coffee maker',
        'Hot water', 'Smoke alarm', 'Essentials', 'Hair dryer', 'Iron', 'Bed linens',
        'Shampoo', 'Free washer'
    ]
    
    parsed_amenities = []
    for amenities_str in data['amenities']:
        if pd.notna(amenities_str):
            try:
                amenities_list = json.loads(amenities_str)
                parsed_amenities.append(set(amenities_list))
            except:
                parsed_amenities.append(set())
        else:
            parsed_amenities.append(set())
    
    for amenity in important_amenities:
        col_name = 'has_' + amenity.lower().replace(' ', '_').replace('-', '_').replace('ö', 'oe').replace('ä', 'ae').replace('ü', 'ue').replace('.', '').replace('/', '_').replace("'", '').replace(':', '').replace('–', '_')
        data[col_name] = [1 if amenity in amenity_set else 0 for amenity_set in parsed_amenities]
    
    data = data.drop(columns=['amenities'])

    # Extract numeric value from bathrooms_text
    data['bathrooms'] = data['bathrooms_text'].apply(lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if pd.notna(x) and re.findall(r'\d+\.?\d*', str(x)) else np.nan)
    data = data.drop(columns=['bathrooms_text'])

    # Convert property_type to binary columns
    property_types = data['property_type'].unique()
    for property_type in property_types:
        if pd.notna(property_type):
            col_name = 'property_' + property_type.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('ö', 'oe').replace('ä', 'ae').replace('ü', 'ue')
            data[col_name] = (data['property_type'] == property_type).astype(int)
    data = data.drop(columns=['property_type'])

    # Impute missing values in numeric columns with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)

    # Clean up column names: remove multiple consecutive underscores
    data.columns = [re.sub(r'_+', '_', col) for col in data.columns]

    return data


if __name__ == '__main__':
    data = preprocess()
    data.to_csv('data/data.csv', index=False)
    