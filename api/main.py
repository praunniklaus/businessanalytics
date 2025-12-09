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
import json
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
models_dir = os.path.join(project_root, 'models')
averages_path = os.path.join(project_root, 'data', 'feature_averages.json')
model_metrics_path = os.path.join(models_dir, 'model_metrics.json')


def load_feature_averages() -> dict:
    try:
        with open(averages_path, "r") as f:
            return json.load(f)
    except Exception:
        pass

    # Fallback: compute basic averages from raw_data if the file is missing
    raw_data_path = os.path.join(project_root, 'data', 'raw_data.csv')
    if os.path.exists(raw_data_path):
        try:
            raw_df = pd.read_csv(raw_data_path)

            host_experience_days = None
            if 'host_experience' in raw_df.columns:
                host_experience_days = float(raw_df['host_experience'].dropna().mean())
            elif 'host_since' in raw_df.columns:
                reference_col = raw_df['last_scraped'] if 'last_scraped' in raw_df.columns else None
                host_since_dt = pd.to_datetime(raw_df['host_since'], errors='coerce')
                if reference_col is not None:
                    ref_dt = pd.to_datetime(reference_col, errors='coerce')
                    host_experience_days = (ref_dt - host_since_dt).dt.days.dropna().mean()
                else:
                    host_experience_days = (pd.Timestamp.today() - host_since_dt).dt.days.dropna().mean()

            return {
                "host_experience_days": float(host_experience_days) if host_experience_days is not None else 365.0,
                "num_listings": float(raw_df['host_total_listings_count'].dropna().mean())
                if 'host_total_listings_count' in raw_df.columns else 1.0,
                "review_scores_rating": float(raw_df['review_scores_rating'].dropna().mean()),
                "review_scores_accuracy": float(raw_df['review_scores_accuracy'].dropna().mean()),
                "review_scores_cleanliness": float(raw_df['review_scores_cleanliness'].dropna().mean()),
                "review_scores_checkin": float(raw_df['review_scores_checkin'].dropna().mean()),
                "review_scores_communication": float(raw_df['review_scores_communication'].dropna().mean()),
                "review_scores_location": float(raw_df['review_scores_location'].dropna().mean()),
                "review_scores_value": float(raw_df['review_scores_value'].dropna().mean()),
                "number_of_reviews": float(raw_df['number_of_reviews'].dropna().mean()),
                "bedrooms": float(raw_df['bedrooms'].dropna().mean()) if 'bedrooms' in raw_df.columns else 1.0,
                "beds": float(raw_df['beds'].dropna().mean()) if 'beds' in raw_df.columns else 1.0,
                "bathrooms": float(raw_df['bathrooms'].dropna().mean()) if 'bathrooms' in raw_df.columns else 1.0,
            }
        except Exception:
            pass

    # Sensible hard fallbacks
    return {
        "host_experience_days": 365.0,
        "num_listings": 1.0,
        "review_scores_rating": 4.7,
        "review_scores_accuracy": 4.7,
        "review_scores_cleanliness": 4.7,
        "review_scores_checkin": 4.7,
        "review_scores_communication": 4.7,
        "review_scores_location": 4.7,
        "review_scores_value": 4.7,
        "number_of_reviews": 50.0,
        "bedrooms": 1.0,
        "beds": 1.0,
        "bathrooms": 1.0,
    }


feature_averages = load_feature_averages()


def evaluate_models_once() -> dict:
    """Evaluate available models and cache RMSE scores."""
    metrics = {}
    data_path = os.path.join(project_root, 'data', 'data.csv')
    if not os.path.exists(data_path):
        return metrics

    data = pd.read_csv(data_path)
    if 'price' not in data.columns:
        return metrics

    X = data.drop(['price', 'id'], axis=1)
    y = data['price']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    for fname in os.listdir(models_dir):
        if not fname.endswith('.joblib'):
            continue
        path = os.path.join(models_dir, fname)
        try:
            model = joblib.load(path)
            preds = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            metrics[fname] = rmse
        except Exception as err:
            print(f"Skipping model {fname} due to error: {err}")
            continue

    if metrics:
        try:
            with open(model_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as err:
            print(f"Failed to write model metrics: {err}")
    return metrics


def load_model_metrics() -> dict:
    if os.path.exists(model_metrics_path):
        try:
            with open(model_metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def select_best_model():
    metrics = load_model_metrics()
    if not metrics:
        metrics = evaluate_models_once()

    if metrics:
        best_name = min(metrics.items(), key=lambda kv: kv[1])[0]
    else:
        # Default to current best-performer
        best_name = 'catboost_tuned.joblib'

    path = os.path.join(models_dir, best_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)
    print(f"Loaded model '{best_name}' for prediction.")
    return model, best_name


model, active_model_name = select_best_model()

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
    bedrooms: Optional[float] = None
    beds: Optional[float] = None
    bathrooms: Optional[float] = None
    amenities: List[str]
    host_experience_days: Optional[int] = None
    num_listings: Optional[int] = None
    review_scores_rating: Optional[float] = None
    review_scores_accuracy: Optional[float] = None
    review_scores_cleanliness: Optional[float] = None
    review_scores_checkin: Optional[float] = None
    review_scores_communication: Optional[float] = None
    review_scores_location: Optional[float] = None
    review_scores_value: Optional[float] = None
    number_of_reviews: Optional[int] = None
    use_averages: bool = False

@app.get("/options")
def get_options():
    return {
        "neighbourhoods": NEIGHBOURHOODS,
        "room_types": ROOM_TYPES,
        "property_types": PROPERTY_TYPES,
        "amenities": AMENITIES
    }

def resolve_with_averages(data: PropertyInput) -> PropertyInput:
    if not data.use_averages:
        return data

    def pick(current, avg_key):
        avg_val = feature_averages.get(avg_key)
        return current if current is not None else avg_val

    filled = data.copy()
    filled.host_experience_days = pick(data.host_experience_days, "host_experience_days")
    filled.num_listings = pick(data.num_listings, "num_listings")
    filled.review_scores_rating = pick(data.review_scores_rating, "review_scores_rating")
    filled.review_scores_accuracy = pick(data.review_scores_accuracy, "review_scores_accuracy")
    filled.review_scores_cleanliness = pick(data.review_scores_cleanliness, "review_scores_cleanliness")
    filled.review_scores_checkin = pick(data.review_scores_checkin, "review_scores_checkin")
    filled.review_scores_communication = pick(data.review_scores_communication, "review_scores_communication")
    filled.review_scores_location = pick(data.review_scores_location, "review_scores_location")
    filled.review_scores_value = pick(data.review_scores_value, "review_scores_value")
    filled.number_of_reviews = pick(data.number_of_reviews, "number_of_reviews")
    return filled


@app.post("/predict")
def predict_price(data: PropertyInput):
    data = resolve_with_averages(data)
    distances = {}
    for landmark_name, (lat, lon) in LANDMARKS.items():
        distances[f'dist_{landmark_name}'] = haversine_distance(
            data.latitude, data.longitude, lat, lon
        )
    
    host_listings = data.num_listings if data.num_listings is not None else feature_averages.get("num_listings", 1)
    is_professional_host = 1 if host_listings > 2 else 0
    
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
    bedrooms_val = data.bedrooms if data.bedrooms is not None else feature_averages.get("bedrooms", 1)
    beds_val = data.beds if data.beds is not None else feature_averages.get("beds", 1)
    bathrooms_val = data.bathrooms if data.bathrooms is not None else feature_averages.get("bathrooms", 1)
    bedroom_density = bedrooms_val / accommodates
    beds_per_person = beds_val / accommodates
    bathrooms_per_person = bathrooms_val / accommodates
    total_rooms = bedrooms_val + bathrooms_val
    
    dist_values = list(distances.values())
    avg_dist_to_landmark = sum(dist_values) / len(dist_values)
    min_dist_to_landmark = min(dist_values)
    
    review_scores = [
        data.review_scores_rating if data.review_scores_rating is not None else feature_averages.get("review_scores_rating", 4.7),
        data.review_scores_accuracy if data.review_scores_accuracy is not None else feature_averages.get("review_scores_accuracy", 4.7),
        data.review_scores_cleanliness if data.review_scores_cleanliness is not None else feature_averages.get("review_scores_cleanliness", 4.7),
        data.review_scores_checkin if data.review_scores_checkin is not None else feature_averages.get("review_scores_checkin", 4.7),
        data.review_scores_communication if data.review_scores_communication is not None else feature_averages.get("review_scores_communication", 4.7),
        data.review_scores_location if data.review_scores_location is not None else feature_averages.get("review_scores_location", 4.7),
        data.review_scores_value if data.review_scores_value is not None else feature_averages.get("review_scores_value", 4.7)
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
        'bedrooms': bedrooms_val,
        'beds': beds_val,
        'review_scores_rating': data.review_scores_rating if data.review_scores_rating is not None else feature_averages.get("review_scores_rating", 4.7),
        'number_of_reviews': data.number_of_reviews if data.number_of_reviews is not None else feature_averages.get("number_of_reviews", 50),
        'review_scores_accuracy': data.review_scores_accuracy if data.review_scores_accuracy is not None else feature_averages.get("review_scores_accuracy", 4.7),
        'review_scores_cleanliness': data.review_scores_cleanliness if data.review_scores_cleanliness is not None else feature_averages.get("review_scores_cleanliness", 4.7),
        'review_scores_checkin': data.review_scores_checkin if data.review_scores_checkin is not None else feature_averages.get("review_scores_checkin", 4.7),
        'review_scores_communication': data.review_scores_communication if data.review_scores_communication is not None else feature_averages.get("review_scores_communication", 4.7),
        'review_scores_location': data.review_scores_location if data.review_scores_location is not None else feature_averages.get("review_scores_location", 4.7),
        'review_scores_value': data.review_scores_value if data.review_scores_value is not None else feature_averages.get("review_scores_value", 4.7),
        'host_experience': data.host_experience_days if data.host_experience_days is not None else feature_averages.get("host_experience_days", 365),
        'is_professional_host': is_professional_host,
        **room_type_features,
        **neighbourhood_features,
        **distances,
        **amenity_features,
        'bathrooms': bathrooms_val,
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
    accommodates: Optional[int] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    rating_min: Optional[float] = Query(None),
    property_type: Optional[str] = Query(None),
    room_type: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
    focus_price: Optional[float] = Query(None),
    focus_bedrooms: Optional[float] = Query(None),
    focus_beds: Optional[float] = Query(None),
    focus_bathrooms: Optional[float] = Query(None),
    focus_accommodates: Optional[int] = Query(None),
):
    blue_marker = None
    if lat is not None and lng is not None:
        blue_marker = {
            'lat': lat,
            'lng': lng,
            'price': price,
            'bedrooms': focus_bedrooms,
            'beds': focus_beds,
            'bathrooms': focus_bathrooms,
            'accommodates': focus_accommodates,
            'neighbourhood': neighbourhood,
            'property_type': property_type,
            'room_type': room_type,
        }
    
    html = create_property_map(
        blue_marker=blue_marker, 
        return_html=True,
        neighbourhood=neighbourhood,
        bedrooms=bedrooms,
        beds=beds,
        accommodates=accommodates,
        price_min=price_min,
        price_max=price_max,
        rating_min=rating_min,
        property_type_filter=property_type,
        room_type_filter=room_type,
        mode=mode,
        focus_lat=lat,
        focus_lng=lng,
        focus_price=price,
        focus_bedrooms=focus_bedrooms,
        focus_beds=focus_beds,
        focus_bathrooms=focus_bathrooms,
        focus_accommodates=focus_accommodates,
    )
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

