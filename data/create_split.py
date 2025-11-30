import pandas as pd
from sklearn.model_selection import train_test_split
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_split_csv():
    data_path = os.path.join(project_root, 'data', 'data.csv')
    data = pd.read_csv(data_path)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=10)
    
    split_df = pd.DataFrame({
        'id': pd.concat([train_data['id'], test_data['id']]),
        'set': ['train'] * len(train_data) + ['test'] * len(test_data)
    })
    
    split_df = split_df.sort_values('id').reset_index(drop=True)
    
    output_path = os.path.join(project_root, 'data', 'split.csv')
    split_df.to_csv(output_path, index=False)
    
    print(f"Split CSV saved to {output_path}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == '__main__':
    create_split_csv()

