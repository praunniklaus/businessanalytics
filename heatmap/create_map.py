import pandas as pd
import folium
from sklearn.model_selection import train_test_split
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_property_map(blue_marker=None, return_html=False, 
                        neighbourhood=None, bedrooms=None, beds=None, accommodates=None):
    data_path = os.path.join(project_root, 'data', 'data.csv')
    raw_data_path = os.path.join(project_root, 'data', 'raw_data.csv')
    
    data = pd.read_csv(data_path)
    raw_data = pd.read_csv(raw_data_path)
    
    train_data, _ = train_test_split(data, test_size=0.2, random_state=10)
    
    train_ids = train_data['id'].values
    
    raw_subset = raw_data[raw_data['id'].isin(train_ids)][
        ['id', 'latitude', 'longitude', 'price', 'review_scores_rating', 
         'property_type', 'bedrooms', 'neighbourhood_group_cleansed',
         'beds', 'accommodates']
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
    
    raw_subset['price_clean'] = raw_subset['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
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
            neighbourhood = row['neighbourhood_group_cleansed'] if pd.notna(row['neighbourhood_group_cleansed']) else 'N/A'
            
            tooltip_text = f"""
            <b>Price:</b> {row['price']}<br>
            <b>Review Score:</b> {review}<br>
            <b>Property Type:</b> {property_type}<br>
            <b>Bedrooms:</b> {bedrooms}<br>
            <b>Neighbourhood:</b> {neighbourhood}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=folium.Tooltip(tooltip_text)
            ).add_to(m)
    
    if blue_marker:
        lat, lng, price = blue_marker['lat'], blue_marker['lng'], blue_marker.get('price', 'N/A')
        tooltip_text = f"""
        <b>YOUR PROPERTY</b><br>
        <b>Predicted Price:</b> ${price}<br>
        <b>Location:</b> {lat:.4f}, {lng:.4f}
        """
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
