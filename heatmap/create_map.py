import os
import pandas as pd
import folium
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

knn_bundle = None


def load_knn_bundle():
    global knn_bundle
    if knn_bundle is not None:
        return knn_bundle

    knn_path = os.path.join(project_root, "models", "knn_similarity.joblib")
    if not os.path.exists(knn_path):
        return None

    try:
        knn_bundle = joblib.load(knn_path)
    except Exception as err:
        print(f"Failed to load KNN bundle: {err}")
        knn_bundle = None
    return knn_bundle


def find_similar_ids(focus_features):
    bundle = load_knn_bundle()
    if not bundle:
        return None

    feature_df = bundle["features"]
    feature_cols = bundle["feature_names"]
    scaler: StandardScaler = bundle["scaler"]
    nn: NearestNeighbors = bundle["nn"]

    if feature_df.empty:
        return None

    row = []
    for col in feature_cols:
        val = focus_features.get(col)
        if val is None:
            val = feature_df[col].median()
        row.append(val)

    vector = scaler.transform([row])
    n_neighbors = min(100, len(feature_df))
    _, indices = nn.kneighbors(vector, n_neighbors=n_neighbors)
    ids = feature_df.iloc[indices[0]]["id"].tolist()
    return set(ids)


def create_property_map(
    blue_marker=None,
    return_html=False,
    neighbourhood=None,
    bedrooms=None,
    beds=None,
    accommodates=None,
    price_min=None,
    price_max=None,
    rating_min=None,
    property_type_filter=None,
    room_type_filter=None,
    mode=None,
    focus_lat=None,
    focus_lng=None,
    focus_price=None,
    focus_bedrooms=None,
    focus_beds=None,
    focus_bathrooms=None,
    focus_accommodates=None,
):
    data_path = os.path.join(project_root, 'data', 'data.csv')
    raw_data_path = os.path.join(project_root, 'data', 'raw_data.csv')
    
    data = pd.read_csv(data_path)
    raw_data = pd.read_csv(raw_data_path)
    
    train_data, _ = train_test_split(data, test_size=0.2, random_state=10)
    
    train_ids = train_data['id'].values
    
    raw_subset = raw_data[raw_data['id'].isin(train_ids)][
        [
            'id',
            'latitude',
            'longitude',
            'price',
            'review_scores_rating',
            'property_type',
            'room_type',
            'bedrooms',
            'neighbourhood_group_cleansed',
            'beds',
            'accommodates',
            'bathrooms_text',
            'number_of_reviews'
        ]
    ].copy()
    
    if neighbourhood:
        # Map API/Frontend names to CSV names
        NEIGHBOURHOOD_MAPPING = {
            'Neukoelln': 'Neukölln',
            'Tempelhof-Schoeneberg': 'Tempelhof - Schöneberg',
            'Treptow-Koepenick': 'Treptow - Köpenick',
            'Steglitz-Zehlendorf': 'Steglitz - Zehlendorf',
            'Marzahn-Hellersdorf': 'Marzahn - Hellersdorf',
            'Charlottenburg-Wilm.': 'Charlottenburg-Wilm.',
            'Friedrichshain-Kreuzberg': 'Friedrichshain-Kreuzberg',
            'Pankow': 'Pankow',
            'Mitte': 'Mitte',
            'Lichtenberg': 'Lichtenberg',
            'Spandau': 'Spandau',
            'Reinickendorf': 'Reinickendorf'
        }
        mapped_neighbourhood = NEIGHBOURHOOD_MAPPING.get(neighbourhood, neighbourhood)
        raw_subset = raw_subset[raw_subset['neighbourhood_group_cleansed'] == mapped_neighbourhood]
    
    if bedrooms is not None:
        raw_subset = raw_subset[raw_subset['bedrooms'] == bedrooms]
        
    if beds is not None:
        raw_subset = raw_subset[raw_subset['beds'] == beds]
        
    if accommodates is not None:
        raw_subset = raw_subset[raw_subset['accommodates'] == accommodates]

    raw_subset['price_clean'] = (
        raw_subset['price']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    if price_min is not None:
        raw_subset = raw_subset[raw_subset['price_clean'] >= price_min]

    if price_max is not None:
        raw_subset = raw_subset[raw_subset['price_clean'] <= price_max]

    if rating_min is not None:
        raw_subset = raw_subset[raw_subset['review_scores_rating'].fillna(0) >= rating_min]

    if property_type_filter:
        raw_subset = raw_subset[raw_subset['property_type'] == property_type_filter]

    if room_type_filter:
        raw_subset = raw_subset[raw_subset['room_type'] == room_type_filter]

    similar_ids = None
    if mode == "advanced" and focus_lat is not None and focus_lng is not None:
        focus_features = {
            "latitude": focus_lat,
            "longitude": focus_lng,
            "accommodates": focus_accommodates,
            "bedrooms": focus_bedrooms,
            "beds": focus_beds,
            "bathrooms_numeric": focus_bathrooms,
            "price_clean": focus_price,
        }
        similar_ids = find_similar_ids(focus_features) or None
        if similar_ids:
            raw_subset = raw_subset[raw_subset['id'].isin(similar_ids)]
    
    low_threshold = raw_subset['price_clean'].quantile(1/3)
    high_threshold = raw_subset['price_clean'].quantile(2/3)
    
    def get_color(price):
        if price <= low_threshold:
            return 'red'
        elif price <= high_threshold:
            return 'orange'
        else:
            return 'green'
    
    berlin_center = [52.52, 13.405]
    m = folium.Map(location=berlin_center, zoom_start=11, tiles='CartoDB positron')
    
    for _, row in raw_subset.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            color = get_color(row['price_clean'])
            
            review = row['review_scores_rating'] if pd.notna(row['review_scores_rating']) else 'N/A'
            bedrooms = int(row['bedrooms']) if pd.notna(row['bedrooms']) else 'N/A'
            property_type = row['property_type'] if pd.notna(row['property_type']) else 'N/A'
            room_type = row['room_type'] if pd.notna(row['room_type']) else 'N/A'
            bathrooms = row['bathrooms_text'] if pd.notna(row['bathrooms_text']) else 'N/A'
            neighbourhood = row['neighbourhood_group_cleansed'] if pd.notna(row['neighbourhood_group_cleansed']) else 'N/A'
            num_reviews = int(row['number_of_reviews']) if pd.notna(row['number_of_reviews']) else 'N/A'
            
            popup_text = f"""
            <div style="font-size: 13px; line-height: 1.4;">
                <b>Price:</b> {row['price']}<br>
                <b>Neighbourhood:</b> {neighbourhood}<br>
                <b>Property type:</b> {property_type}<br>
                <b>Room type:</b> {room_type}<br>
                <b>Bedrooms/Beds/Baths:</b> {bedrooms} / {row['beds']} / {bathrooms}<br>
                <b>Accommodates:</b> {row['accommodates']} guests<br>
                <b>Review score:</b> {review} ({num_reviews} reviews)<br>
                <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=320),
                tooltip=folium.Tooltip(f"{property_type} • {row['price']} • {neighbourhood}")
            ).add_to(m)
    
    if blue_marker:
        lat, lng, price = blue_marker['lat'], blue_marker['lng'], blue_marker.get('price', 'N/A')
        detail_accommodates = blue_marker.get('accommodates')
        detail_bedrooms = blue_marker.get('bedrooms')
        detail_beds = blue_marker.get('beds')
        detail_bathrooms = blue_marker.get('bathrooms')
        detail_neighbourhood = blue_marker.get('neighbourhood')
        detail_property_type = blue_marker.get('property_type')
        detail_room_type = blue_marker.get('room_type')

        tooltip_lines = [
            "<b>YOUR PROPERTY</b>",
            f"<b>Predicted Price:</b> ${price}",
            f"<b>Location:</b> {lat:.4f}, {lng:.4f}",
        ]
        if detail_neighbourhood:
            tooltip_lines.append(f"<b>Neighbourhood:</b> {detail_neighbourhood}")
        if detail_property_type:
            tooltip_lines.append(f"<b>Property type:</b> {detail_property_type}")
        if detail_room_type:
            tooltip_lines.append(f"<b>Room type:</b> {detail_room_type}")
        if detail_accommodates is not None:
            tooltip_lines.append(f"<b>Accommodates:</b> {detail_accommodates}")
        if detail_bedrooms is not None or detail_beds is not None or detail_bathrooms is not None:
            tooltip_lines.append(
                f"<b>Bedrooms/Beds/Baths:</b> {detail_bedrooms or 'N/A'} / {detail_beds or 'N/A'} / {detail_bathrooms or 'N/A'}"
            )
        tooltip_text = "<br>".join(tooltip_lines)
        folium.CircleMarker(
            location=[lat, lng],
            radius=10,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip_text)
        ).add_to(m)
        m.location = [lat, lng]
        m.zoom_start = 14
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; font-size: 14px;">
        <b>Price Category</b><br>
        <i style="background: red; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Bottom 33% (Low)<br>
        <i style="background: orange; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Middle 33%<br>
        <i style="background: green; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Top 33% (High)<br>
        <i style="background: blue; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Your Property
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Send click events to parent window so the frontend can auto-fill coordinates
    click_js = f"""
    <script>
    (function() {{
        var map = {m.get_name()};
        map.on('click', function(e) {{
            var payload = {{
                type: "map-click",
                lat: e.latlng.lat,
                lng: e.latlng.lng
            }};
            try {{
                window.parent.postMessage(payload, window.location.origin);
            }} catch (err) {{
                console.error('Map postMessage failed', err);
            }}
        }});
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(click_js))
    
    if return_html:
        return m._repr_html_()
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'property_map.html')
    m.save(output_path)
    print(f"Map saved to {output_path}")
    print(f"Total properties displayed: {len(raw_subset)}")
    print(f"Price thresholds: Low <= ${low_threshold:.2f}, Middle <= ${high_threshold:.2f}, High > ${high_threshold:.2f}")

if __name__ == '__main__':
    create_property_map()
