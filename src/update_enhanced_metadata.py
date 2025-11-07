"""
Complete Enhanced Dataset Metadata Update
=========================================
Updates the enhanced dataset metadata to include all urgency levels:
- HIGH: Real fire/emergency images from Kaggle
- MEDIUM: Synthetic civic issues  
- LOW: Synthetic civic issues
"""

import pandas as pd
from pathlib import Path
import json

# Load the current enhanced metadata (only has HIGH priority)
enhanced_dir = Path("data/images_enhanced")
metadata_path = enhanced_dir / "metadata.csv"

# Load existing high priority fire images metadata
df_high = pd.read_csv(metadata_path)
print(f"📊 Current HIGH priority images: {len(df_high)}")

# Load original synthetic metadata for MEDIUM and LOW
original_metadata_path = Path("data/images_simple/metadata.csv")
if original_metadata_path.exists():
    df_original = pd.read_csv(original_metadata_path)
    
    # Get MEDIUM and LOW priority synthetic images
    df_medium_low = df_original[df_original['urgency_level'].isin(['MEDIUM', 'LOW'])]
    
    # Update file paths to point to enhanced dataset location
    df_medium_low = df_medium_low.copy()
    df_medium_low['filepath'] = df_medium_low['filepath'].str.replace(
        '..\\data\\images_simple\\', 
        '..\\data\\images_enhanced\\',
        regex=False
    )
    
    print(f"📊 MEDIUM priority images: {len(df_medium_low[df_medium_low['urgency_level'] == 'MEDIUM'])}")
    print(f"📊 LOW priority images: {len(df_medium_low[df_medium_low['urgency_level'] == 'LOW'])}")
    
    # Combine all metadata
    df_complete = pd.concat([df_high, df_medium_low], ignore_index=True)
    
    # Save complete enhanced metadata
    df_complete.to_csv(metadata_path, index=False)
    
    print(f"✅ Complete enhanced metadata saved")
    print(f"📊 Total enhanced dataset: {len(df_complete)} images")
    print(f"   🔥 HIGH (Fire/Emergency): {len(df_complete[df_complete['urgency_level'] == 'HIGH'])}")
    print(f"   ⚠️ MEDIUM (Synthetic): {len(df_complete[df_complete['urgency_level'] == 'MEDIUM'])}")
    print(f"   📝 LOW (Synthetic): {len(df_complete[df_complete['urgency_level'] == 'LOW'])}")
    
    # Update summary
    summary = {
        'dataset_type': 'enhanced_with_fire_images',
        'total_images': len(df_complete),
        'high_priority_source': 'kaggle_fire_dataset',
        'medium_low_priority_source': 'synthetic_generated',
        'urgency_distribution': df_complete['urgency_level'].value_counts().to_dict(),
        'created_date': pd.Timestamp.now().isoformat(),
        'fire_dataset_info': {
            'source': 'phylake1337/fire-dataset',
            'fire_images_used': len(df_high),
            'total_fire_images_available': 755,
            'dataset_size': '387MB'
        }
    }
    
    summary_path = enhanced_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ Enhanced dataset summary updated")
    
else:
    print("⚠️ Original synthetic metadata not found")