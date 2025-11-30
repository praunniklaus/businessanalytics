import pandas as pd
import numpy as np

def engineer_features():
    data = pd.read_csv('data/preprocessed.csv')
    
    # Bedroom density (space per person)
    data['bedroom_density'] = data['bedrooms'] / data['accommodates'].replace(0, 1)
    
    # Beds per person
    data['beds_per_person'] = data['beds'] / data['accommodates'].replace(0, 1)
    
    # Bathrooms per person
    data['bathrooms_per_person'] = data['bathrooms'] / data['accommodates'].replace(0, 1)
    
    # Total rooms (bedrooms + bathrooms)
    data['total_rooms'] = data['bedrooms'] + data['bathrooms']
    
    # Average distance to landmarks (complement to min distance)
    dist_cols = [col for col in data.columns if col.startswith('dist_')]
    data['avg_dist_to_landmark'] = data[dist_cols].mean(axis=1)
    
    # Review score consistency (std of review scores)
    review_cols = ['review_scores_rating', 'review_scores_accuracy', 
                   'review_scores_cleanliness', 'review_scores_checkin',
                   'review_scores_communication', 'review_scores_location', 
                   'review_scores_value']
    data['review_score_std'] = data[review_cols].std(axis=1)
    
    # Minimum distance to any landmark (proximity to city center)
    data['min_dist_to_landmark'] = data[dist_cols].min(axis=1)
    
    # Total amenities count
    amenity_cols = [col for col in data.columns if col.startswith('has_')]
    data['total_amenities'] = data[amenity_cols].sum(axis=1)
    
    data.to_csv('data/data.csv', index=False)
    print(f"Feature engineering complete. Output saved to data/data.csv")
    print(f"Added {len(['bedroom_density', 'beds_per_person', 'bathrooms_per_person', 
                        'total_rooms', 'avg_dist_to_landmark', 'review_score_std', 
                        'min_dist_to_landmark', 'total_amenities'])} new features.")

if __name__ == '__main__':
    engineer_features()

