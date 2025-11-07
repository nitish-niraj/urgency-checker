"""
Enhanced Dataset Integration - Fire Dataset for HIGH Priority
============================================================
This script downloads real fire/emergency images from Kaggle and integrates them
with the existing civic issue urgency classifier for more realistic HIGH priority classification.
"""

import kagglehub
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import json

class EnhancedDatasetManager:
    """Manages integration of real fire dataset with synthetic civic issue data"""
    
    def __init__(self, base_data_dir="../data"):
        self.base_data_dir = Path(base_data_dir)
        self.images_dir = self.base_data_dir / "images_enhanced"
        self.fire_dataset_path = None
        
    def download_fire_dataset(self):
        """Download the fire dataset from Kaggle"""
        print("🔥 Downloading Fire Dataset from Kaggle...")
        print("=" * 50)
        
        try:
            # Download latest version
            path = kagglehub.dataset_download("phylake1337/fire-dataset")
            print(f"✅ Dataset downloaded to: {path}")
            self.fire_dataset_path = Path(path)
            return path
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("💡 Make sure you have Kaggle API credentials configured:")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Create new API token")
            print("   3. Place kaggle.json in ~/.kaggle/ directory")
            return None
    
    def explore_fire_dataset(self):
        """Explore the downloaded fire dataset structure"""
        if not self.fire_dataset_path:
            print("❌ Fire dataset not downloaded yet")
            return
            
        print("\n🔍 Exploring Fire Dataset Structure...")
        print("=" * 40)
        
        def explore_directory(path, level=0):
            """Recursively explore directory structure"""
            indent = "  " * level
            if path.is_file():
                size = path.stat().st_size / (1024*1024)  # MB
                print(f"{indent}📄 {path.name} ({size:.2f} MB)")
            else:
                print(f"{indent}📁 {path.name}/")
                try:
                    items = sorted(path.iterdir())
                    for item in items[:10]:  # Limit to first 10 items
                        explore_directory(item, level + 1)
                    if len(items) > 10:
                        print(f"{indent}  ... and {len(items) - 10} more items")
                except PermissionError:
                    print(f"{indent}  (Permission denied)")
        
        explore_directory(self.fire_dataset_path)
    
    def process_fire_images_for_high_priority(self, max_images=50):
        """Process fire images to use as HIGH priority civic emergency samples"""
        if not self.fire_dataset_path:
            print("❌ Fire dataset not available")
            return
            
        print(f"\n🔥 Processing Fire Images for HIGH Priority Classification...")
        print("=" * 60)
        
        # Create enhanced images directory
        self.images_dir.mkdir(parents=True, exist_ok=True)
        high_dir = self.images_dir / "HIGH"
        high_dir.mkdir(exist_ok=True)
        
        # Find fire/emergency images
        fire_images = []
        
        # Common fire dataset patterns
        fire_patterns = ['fire', 'Fire', 'FIRE', 'emergency', 'Emergency']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        def find_fire_images(directory):
            """Recursively find fire images"""
            images_found = []
            try:
                for item in directory.rglob('*'):
                    if item.is_file() and item.suffix.lower() in image_extensions:
                        # Check if filename or parent directory suggests fire/emergency
                        item_str = str(item).lower()
                        if any(pattern.lower() in item_str for pattern in fire_patterns):
                            images_found.append(item)
                        # Also include images from directories that suggest emergencies
                        elif any(word in item.parent.name.lower() for word in ['emergency', 'urgent', 'critical', 'danger']):
                            images_found.append(item)
            except Exception as e:
                print(f"⚠️ Error exploring {directory}: {e}")
            return images_found
        
        print("🔍 Searching for fire/emergency images...")
        fire_images = find_fire_images(self.fire_dataset_path)
        
        if not fire_images:
            print("⚠️ No fire images found with expected patterns")
            print("📁 Listing all image files in dataset:")
            all_images = list(self.fire_dataset_path.rglob('*.jpg')) + \
                        list(self.fire_dataset_path.rglob('*.jpeg')) + \
                        list(self.fire_dataset_path.rglob('*.png'))
            
            for img in all_images[:20]:  # Show first 20
                print(f"   📸 {img.relative_to(self.fire_dataset_path)}")
            
            if len(all_images) > 20:
                print(f"   ... and {len(all_images) - 20} more images")
            
            # Use all images as potential emergency scenarios
            fire_images = all_images[:max_images]
        
        print(f"📸 Found {len(fire_images)} potential emergency images")
        
        # Process and copy fire images
        processed_images = []
        metadata_entries = []
        
        for i, fire_img_path in enumerate(fire_images[:max_images]):
            try:
                # Load and process image
                with Image.open(fire_img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to standard size (224x224 for consistency)
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Save with consistent naming
                    output_filename = f"fire_emergency_{i:03d}.jpg"
                    output_path = high_dir / output_filename
                    
                    img_resized.save(output_path, 'JPEG', quality=85)
                    processed_images.append(output_path)
                    
                    # Create metadata entry
                    metadata_entries.append({
                        'filepath': str(output_path.relative_to(self.base_data_dir)),
                        'urgency_level': 'HIGH',
                        'urgency_score': 10,  # Maximum urgency for fire/emergency
                        'description': 'Emergency fire/safety hazard requiring immediate response',
                        'category': 'Emergency',
                        'source': 'fire_dataset',
                        'original_file': str(fire_img_path.name)
                    })
                    
                    if (i + 1) % 10 == 0:
                        print(f"   ✅ Processed {i + 1}/{min(len(fire_images), max_images)} images")
                        
            except Exception as e:
                print(f"   ⚠️ Error processing {fire_img_path.name}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(processed_images)} fire/emergency images")
        
        # Create metadata file for the enhanced dataset
        self.create_enhanced_metadata(metadata_entries)
        
        return processed_images
    
    def create_enhanced_metadata(self, fire_metadata):
        """Create enhanced metadata combining fire images with existing synthetic data"""
        print("\n📋 Creating Enhanced Dataset Metadata...")
        
        # Load existing synthetic metadata if it exists
        existing_metadata = []
        old_metadata_path = self.base_data_dir / "images_simple" / "metadata.csv"
        
        if old_metadata_path.exists():
            try:
                df_existing = pd.read_csv(old_metadata_path)
                # Filter out HIGH priority synthetic images (replace with real fire images)
                df_existing = df_existing[df_existing['urgency_level'] != 'HIGH']
                existing_metadata = df_existing.to_dict('records')
                print(f"📊 Kept {len(existing_metadata)} MEDIUM/LOW synthetic images")
            except Exception as e:
                print(f"⚠️ Error loading existing metadata: {e}")
        
        # Combine fire images (HIGH) with existing MEDIUM/LOW synthetic images
        all_metadata = fire_metadata + existing_metadata
        
        # Create enhanced metadata DataFrame
        df_enhanced = pd.DataFrame(all_metadata)
        
        # Save enhanced metadata
        enhanced_metadata_path = self.images_dir / "metadata.csv"
        df_enhanced.to_csv(enhanced_metadata_path, index=False)
        
        print(f"✅ Enhanced metadata saved: {enhanced_metadata_path}")
        print(f"📊 Total images in enhanced dataset: {len(all_metadata)}")
        print(f"   🔥 HIGH (Fire/Emergency): {len(fire_metadata)}")
        print(f"   ⚠️ MEDIUM (Synthetic): {len([m for m in existing_metadata if m.get('urgency_level') == 'MEDIUM'])}")
        print(f"   📝 LOW (Synthetic): {len([m for m in existing_metadata if m.get('urgency_level') == 'LOW'])}")
        
        # Save summary statistics
        summary = {
            'dataset_type': 'enhanced_with_fire_images',
            'total_images': len(all_metadata),
            'high_priority_source': 'kaggle_fire_dataset',
            'medium_low_priority_source': 'synthetic_generated',
            'urgency_distribution': df_enhanced['urgency_level'].value_counts().to_dict(),
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        summary_path = self.images_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Dataset summary saved: {summary_path}")
        
        return df_enhanced
    
    def update_image_classifier_config(self):
        """Update the image classifier to use the enhanced dataset"""
        print("\n🔧 Updating Image Classifier Configuration...")
        
        # Update the image classifier script to point to enhanced dataset
        classifier_script = Path("src/image_classifier.py")
        
        if classifier_script.exists():
            print("📝 Image classifier will need to be retrained with enhanced dataset")
            print("💡 The enhanced dataset path is: data/images_enhanced/")
            print("🔄 To retrain: Run the image classifier with the new dataset path")
        
        return str(self.images_dir)

def main():
    """Main function to download and integrate fire dataset"""
    print("🏛️ ENHANCING CIVIC ISSUE URGENCY CLASSIFIER")
    print("🔥 Integrating Real Fire/Emergency Images for HIGH Priority")
    print("=" * 65)
    
    # Initialize dataset manager
    manager = EnhancedDatasetManager()
    
    # Download fire dataset
    dataset_path = manager.download_fire_dataset()
    
    if dataset_path:
        # Explore dataset structure
        manager.explore_fire_dataset()
        
        # Process fire images for HIGH priority
        processed_images = manager.process_fire_images_for_high_priority(max_images=30)
        
        if processed_images:
            # Update classifier configuration
            enhanced_path = manager.update_image_classifier_config()
            
            print(f"\n🎉 DATASET ENHANCEMENT COMPLETE!")
            print("=" * 40)
            print("✅ Real fire/emergency images integrated for HIGH priority")
            print("✅ Synthetic images retained for MEDIUM/LOW priority")
            print("✅ Enhanced metadata created")
            print(f"📁 Enhanced dataset location: {enhanced_path}")
            print("\n🚀 Next Steps:")
            print("1. Retrain the image classifier with enhanced dataset")
            print("2. Test the API with real emergency images")
            print("3. Deploy the improved system")

if __name__ == "__main__":
    main()