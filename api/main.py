from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import os
import sys
import math

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from heatmap.create_map import create_property_map

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'models', 'xgboost.joblib')
model = joblib.load(model_path)

LANDMARKS = {
    'brandenburger_tor': (52.516275, 13.377704),
    'alexanderplatz': (52.521918, 13.413215),
    'reichstagsgebaeude': (52.518620, 13.376187),
    'potsdamer_platz': (52.509168, 13.376641),
    'siegessaeule': (52.514543, 13.350119),
    'tiergarten': (52.509778, 13.357260),
}

NEIGHBOURHOODS = [
    'Pankow', 'Neukoelln', 'Mitte', 'Friedrichshain-Kreuzberg',
    'Charlottenburg-Wilm.', 'Tempelhof-Schoeneberg', 'Lichtenberg',
    'Steglitz-Zehlendorf', 'Treptow-Koepenick', 'Reinickendorf',
    'Marzahn-Hellersdorf', 'Spandau'
]

ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']

PROPERTY_TYPES = [
    'Entire rental unit', 'Private room in rental unit', 'Entire condo',
    'Entire serviced apartment', 'Room in hotel', 'Private room in condo',
    'Entire loft', 'Private room in home', 'Entire home',
    'Private room in bed and breakfast', 'Private room in hostel',
    'Entire guesthouse', 'Room in boutique hotel', 'Shared room in hostel',
    'Room in aparthotel', 'Other'
]

AMENITIES = [
    'Wifi', 'Kitchen', 'Air conditioning', 'Free parking', 'Washer', 'Dryer',
    'Heating', 'TV', 'Dishwasher', 'Refrigerator', 'Microwave', 'Coffee maker',
    'Hot water', 'Smoke alarm', 'Essentials', 'Hair dryer', 'Iron', 'Bed linens',
    'Shampoo', 'Free washer'
]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class PropertyInput(BaseModel):
    latitude: float
    longitude: float
    neighbourhood: str
    room_type: str
    property_type: str
    accommodates: int
    bedrooms: float
    beds: float
    bathrooms: float
    amenities: List[str]
    host_experience_days: int
    num_listings: int
    review_scores_rating: float
    review_scores_accuracy: float
    review_scores_cleanliness: float
    review_scores_checkin: float
    review_scores_communication: float
    review_scores_location: float
    review_scores_value: float
    number_of_reviews: int

@app.get("/options")
def get_options():
    return {
        "neighbourhoods": NEIGHBOURHOODS,
        "room_types": ROOM_TYPES,
        "property_types": PROPERTY_TYPES,
        "amenities": AMENITIES
    }

@app.post("/predict")
def predict_price(data: PropertyInput):
    distances = {}
    for landmark_name, (lat, lon) in LANDMARKS.items():
        distances[f'dist_{landmark_name}'] = haversine_distance(
            data.latitude, data.longitude, lat, lon
        )
    
    is_professional_host = 1 if data.num_listings > 2 else 0
    
    neighbourhood_features = {f'is_{n.lower().replace(" ", "_").replace("-", "_").replace(".", "").replace("ö", "oe").replace("ä", "ae").replace("ü", "ue")}': 0 for n in NEIGHBOURHOODS}
    neighbourhood_key = f'is_{data.neighbourhood.lower().replace(" ", "_").replace("-", "_").replace(".", "").replace("ö", "oe").replace("ä", "ae").replace("ü", "ue")}'
    if neighbourhood_key in neighbourhood_features:
        neighbourhood_features[neighbourhood_key] = 1
    
    room_type_features = {
        'is_entire_home_apt': 0,
        'is_private_room': 0,
        'is_shared_room': 0,
        'is_hotel_room': 0
    }
    room_type_map = {
        'Entire home/apt': 'is_entire_home_apt',
        'Private room': 'is_private_room',
        'Shared room': 'is_shared_room',
        'Hotel room': 'is_hotel_room'
    }
    if data.room_type in room_type_map:
        room_type_features[room_type_map[data.room_type]] = 1
    
    property_type_features = {}
    for pt in PROPERTY_TYPES[:-1]:
        key = 'property_' + pt.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        property_type_features[key] = 1 if data.property_type == pt else 0
    property_type_features['property_other'] = 1 if data.property_type == 'Other' else 0
    
    amenity_features = {}
    for amenity in AMENITIES:
        key = 'has_' + amenity.lower().replace(' ', '_').replace('-', '_')
        amenity_features[key] = 1 if amenity in data.amenities else 0
    
    accommodates = max(data.accommodates, 1)
    bedroom_density = data.bedrooms / accommodates
    beds_per_person = data.beds / accommodates
    bathrooms_per_person = data.bathrooms / accommodates
    total_rooms = data.bedrooms + data.bathrooms
    
    dist_values = list(distances.values())
    avg_dist_to_landmark = sum(dist_values) / len(dist_values)
    min_dist_to_landmark = min(dist_values)
    
    review_scores = [
        data.review_scores_rating, data.review_scores_accuracy,
        data.review_scores_cleanliness, data.review_scores_checkin,
        data.review_scores_communication, data.review_scores_location,
        data.review_scores_value
    ]
    review_score_std = np.std(review_scores)
    
    total_amenities = sum(amenity_features.values())
    
    feature_order = [
        'accommodates', 'bedrooms', 'beds', 'review_scores_rating', 'number_of_reviews',
        'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location', 'review_scores_value',
        'host_experience', 'is_professional_host',
        'is_entire_home_apt', 'is_private_room', 'is_shared_room', 'is_hotel_room',
        'is_pankow', 'is_neukoelln', 'is_mitte', 'is_friedrichshain_kreuzberg',
        'is_charlottenburg_wilm', 'is_tempelhof_schoeneberg', 'is_lichtenberg',
        'is_steglitz_zehlendorf', 'is_treptow_koepenick', 'is_reinickendorf',
        'is_marzahn_hellersdorf', 'is_spandau',
        'dist_brandenburger_tor', 'dist_alexanderplatz', 'dist_reichstagsgebaeude',
        'dist_potsdamer_platz', 'dist_siegessaeule', 'dist_tiergarten',
        'has_wifi', 'has_kitchen', 'has_air_conditioning', 'has_free_parking',
        'has_washer', 'has_dryer', 'has_heating', 'has_tv', 'has_dishwasher',
        'has_refrigerator', 'has_microwave', 'has_coffee_maker', 'has_hot_water',
        'has_smoke_alarm', 'has_essentials', 'has_hair_dryer', 'has_iron',
        'has_bed_linens', 'has_shampoo', 'has_free_washer',
        'bathrooms',
        'property_entire_rental_unit', 'property_private_room_in_rental_unit',
        'property_entire_condo', 'property_entire_serviced_apartment',
        'property_room_in_hotel', 'property_private_room_in_condo',
        'property_entire_loft', 'property_private_room_in_home', 'property_entire_home',
        'property_private_room_in_bed_and_breakfast', 'property_private_room_in_hostel',
        'property_entire_guesthouse', 'property_room_in_boutique_hotel',
        'property_shared_room_in_hostel', 'property_room_in_aparthotel', 'property_other',
        'bedroom_density', 'beds_per_person', 'bathrooms_per_person', 'total_rooms',
        'avg_dist_to_landmark', 'review_score_std', 'min_dist_to_landmark', 'total_amenities'
    ]
    
    features = {
        'accommodates': data.accommodates,
        'bedrooms': data.bedrooms,
        'beds': data.beds,
        'review_scores_rating': data.review_scores_rating,
        'number_of_reviews': data.number_of_reviews,
        'review_scores_accuracy': data.review_scores_accuracy,
        'review_scores_cleanliness': data.review_scores_cleanliness,
        'review_scores_checkin': data.review_scores_checkin,
        'review_scores_communication': data.review_scores_communication,
        'review_scores_location': data.review_scores_location,
        'review_scores_value': data.review_scores_value,
        'host_experience': data.host_experience_days,
        'is_professional_host': is_professional_host,
        **room_type_features,
        **neighbourhood_features,
        **distances,
        **amenity_features,
        'bathrooms': data.bathrooms,
        **property_type_features,
        'bedroom_density': bedroom_density,
        'beds_per_person': beds_per_person,
        'bathrooms_per_person': bathrooms_per_person,
        'total_rooms': total_rooms,
        'avg_dist_to_landmark': avg_dist_to_landmark,
        'review_score_std': review_score_std,
        'min_dist_to_landmark': min_dist_to_landmark,
        'total_amenities': total_amenities,
    }
    
    feature_vector = [features.get(f, 0) for f in feature_order]
    
    prediction = model.predict([feature_vector])[0]
    
    return {
        "predicted_price": round(float(prediction), 2),
        "latitude": data.latitude,
        "longitude": data.longitude
    }

@app.get("/map", response_class=HTMLResponse)
def get_map(
    lat: Optional[float] = Query(None),
    lng: Optional[float] = Query(None),
    price: Optional[float] = Query(None),
    neighbourhood: Optional[str] = Query(None),
    bedrooms: Optional[float] = Query(None),
    beds: Optional[float] = Query(None),
    accommodates: Optional[int] = Query(None)
):
    blue_marker = None
    if lat is not None and lng is not None:
        blue_marker = {'lat': lat, 'lng': lng, 'price': price}
    
    html = create_property_map(
        blue_marker=blue_marker, 
        return_html=True,
        neighbourhood=neighbourhood,
        bedrooms=bedrooms,
        beds=beds,
        accommodates=accommodates
    )
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

