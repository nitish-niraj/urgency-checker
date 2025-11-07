"""
Civic Issue Urgency Classifier - Synthetic Dataset Generator
Creates comprehensive training data for multimodal AI classification system
"""

import pandas as pd
import random
import json
from datetime import datetime, timedelta
import os

class CivicIssueDatasetGenerator:
    def __init__(self):
        self.categories = {
            'road_problems': {
                'issues': ['potholes', 'cracks', 'waterlogging', 'broken roads', 'road collapse', 'uneven surface'],
                'locations': ['main road', 'residential street', 'highway', 'side road', 'intersection', 'bridge']
            },
            'drainage': {
                'issues': ['overflowing drains', 'blocked sewers', 'water stagnation', 'manholes', 'flooding', 'clogged gutters'],
                'locations': ['street corner', 'residential area', 'market area', 'near school', 'hospital vicinity', 'park area']
            },
            'electricity': {
                'issues': ['broken poles', 'exposed wires', 'street light failures', 'transformer problems', 'power outage', 'sparking cables'],
                'locations': ['residential colony', 'commercial area', 'industrial zone', 'school premises', 'hospital area', 'public park']
            },
            'garbage': {
                'issues': ['overflowing bins', 'illegal dumping', 'waste accumulation', 'scattered litter', 'burning garbage', 'medical waste'],
                'locations': ['residential area', 'market place', 'bus stop', 'park', 'school area', 'hospital zone']
            },
            'water_supply': {
                'issues': ['pipe leaks', 'no water supply', 'contaminated water', 'low pressure', 'burst pipes', 'dirty water tanks'],
                'locations': ['residential complex', 'apartment building', 'slum area', 'government office', 'school', 'hospital']
            },
            'public_safety': {
                'issues': ['damaged infrastructure', 'security issues', 'broken railings', 'unsafe structures', 'missing signs', 'hazardous conditions'],
                'locations': ['public building', 'bridge', 'flyover', 'park', 'bus station', 'market area']
            }
        }
        
        self.urgency_levels = {
            'HIGH': {
                'score_range': (8, 10),
                'keywords': ['emergency', 'urgent', 'dangerous', 'hazardous', 'immediate', 'critical', 'life-threatening', 'serious'],
                'time_indicators': ['right now', 'immediately', 'asap', 'today', 'cannot wait']
            },
            'MEDIUM': {
                'score_range': (4, 7),
                'keywords': ['important', 'concerning', 'needs attention', 'problem', 'issue', 'troublesome'],
                'time_indicators': ['soon', 'this week', 'few days', 'when possible', 'at earliest']
            },
            'LOW': {
                'score_range': (1, 3),
                'keywords': ['minor', 'small', 'cosmetic', 'eventually', 'not urgent', 'whenever'],
                'time_indicators': ['sometime', 'no rush', 'when convenient', 'next month', 'eventually']
            }
        }
        
        self.emotional_tones = {
            'formal': ['I would like to report', 'There is a', 'I am writing to inform', 'Please address'],
            'informal': ['Hey there\'s a', 'Can someone fix', 'There\'s this', 'Help! We have'],
            'frustrated': ['I\'m sick of', 'This is ridiculous', 'How long do we wait', 'Fed up with'],
            'concerned': ['I\'m worried about', 'This concerns me', 'I fear that', 'This could be dangerous']
        }
        
        self.location_contexts = {
            'residential': ['in our neighborhood', 'near my house', 'in the residential area', 'on our street'],
            'commercial': ['in the market area', 'near shops', 'in the business district', 'commercial zone'],
            'institutional': ['near the hospital', 'close to school', 'government office area', 'near the clinic'],
            'public': ['in the park', 'at the bus stop', 'public area', 'community center']
        }

    def generate_high_urgency_description(self, category, issue, location):
        """Generate high urgency civic issue descriptions"""
        templates = [
            f"URGENT: {issue} causing serious safety hazard {location}. {random.choice(self.urgency_levels['HIGH']['keywords']).title()} action needed {random.choice(self.urgency_levels['HIGH']['time_indicators'])}.",
            f"Emergency situation: {issue} {location} creating {random.choice(['life-threatening', 'dangerous', 'hazardous'])} conditions. Immediate response required.",
            f"Critical issue - {issue} {location}. This is {random.choice(['extremely dangerous', 'a serious safety risk', 'life-threatening'])}. Please address immediately.",
            f"DANGER: {issue} {location} poses immediate threat to public safety. {random.choice(self.urgency_levels['HIGH']['keywords']).title()} intervention needed."
        ]
        
        base_text = random.choice(templates)
        
        # Add specific high-urgency context based on category
        if category == 'electricity':
            base_text += f" Risk of electrocution. {random.choice(['Children play here', 'Heavy foot traffic area', 'Near water source'])}."
        elif category == 'road_problems':
            base_text += f" Causing accidents. {random.choice(['Vehicles getting damaged', 'Pedestrians at risk', 'Traffic chaos'])}."
        elif category == 'drainage':
            base_text += f" Health hazard. {random.choice(['Disease risk', 'Contamination spreading', 'Mosquito breeding'])}."
        elif category == 'public_safety':
            base_text += f" Someone could get seriously injured. {random.choice(['Structure unstable', 'Sharp edges exposed', 'No warning signs'])}."
        
        return base_text

    def generate_medium_urgency_description(self, category, issue, location):
        """Generate medium urgency civic issue descriptions"""
        templates = [
            f"There's a {issue} {location} that needs attention. It's {random.choice(self.urgency_levels['MEDIUM']['keywords'])} and should be fixed {random.choice(self.urgency_levels['MEDIUM']['time_indicators'])}.",
            f"Reporting {issue} {location}. This is causing {random.choice(['inconvenience', 'problems', 'difficulties'])} for residents. Please address when possible.",
            f"Issue with {issue} {location}. While not {random.choice(['emergency', 'critical'])}, it needs proper attention {random.choice(self.urgency_levels['MEDIUM']['time_indicators'])}.",
            f"Problem: {issue} {location}. This is {random.choice(['troublesome', 'concerning', 'problematic'])} and requires resolution."
        ]
        
        base_text = random.choice(templates)
        
        # Add specific medium-urgency context
        if category == 'garbage':
            base_text += f" {random.choice(['Bad smell', 'Attracting flies', 'Unsightly appearance'])}."
        elif category == 'water_supply':
            base_text += f" {random.choice(['Affecting daily routine', 'Inconvenient for families', 'Disrupting normal activities'])}."
        elif category == 'electricity':
            base_text += f" {random.choice(['Dark streets at night', 'Security concerns', 'Visibility issues'])}."
        
        return base_text

    def generate_low_urgency_description(self, category, issue, location):
        """Generate low urgency civic issue descriptions"""
        templates = [
            f"Minor {issue} {location}. Not {random.choice(self.urgency_levels['LOW']['keywords'])} but could be fixed {random.choice(self.urgency_levels['LOW']['time_indicators'])}.",
            f"Small issue with {issue} {location}. {random.choice(['Cosmetic problem', 'Minor inconvenience', 'Not serious'])} but worth mentioning.",
            f"Noticed {issue} {location}. It's {random.choice(self.urgency_levels['LOW']['keywords'])} but thought I'd report it.",
            f"{issue.title()} {location}. {random.choice(['Eventually needs fixing', 'When convenient', 'No immediate rush'])}."
        ]
        
        base_text = random.choice(templates)
        
        # Add specific low-urgency context
        base_text += f" {random.choice(['Aesthetic issue mainly', 'Not affecting daily life much', 'Could wait for routine maintenance', 'Part of regular upkeep needed'])}"
        
        return base_text

    def generate_image_description(self, category, urgency, issue, location):
        """Generate corresponding image descriptions"""
        base_descriptions = {
            'road_problems': {
                'HIGH': f"Image shows severe {issue} with {random.choice(['visible damage to vehicles', 'people avoiding the area', 'warning cones placed', 'deep holes visible'])}",
                'MEDIUM': f"Photo of {issue} showing {random.choice(['moderate damage', 'some inconvenience to traffic', 'visible but manageable problem'])}",
                'LOW': f"Picture shows minor {issue} with {random.choice(['slight surface damage', 'barely visible cracks', 'cosmetic issues only'])}"
            },
            'drainage': {
                'HIGH': f"Image shows {issue} with {random.choice(['water flooding streets', 'sewage spillover', 'people wading through water', 'health hazard visible'])}",
                'MEDIUM': f"Photo of {issue} showing {random.choice(['standing water', 'clogged drain visible', 'water accumulation'])}",
                'LOW': f"Picture shows minor {issue} with {random.choice(['slight water pooling', 'small blockage', 'minimal water collection'])}"
            },
            'electricity': {
                'HIGH': f"Image shows {issue} with {random.choice(['exposed live wires', 'sparking visible', 'dangerous electrical hazard', 'people staying away'])}",
                'MEDIUM': f"Photo of {issue} showing {random.choice(['non-functioning equipment', 'dark streets', 'electrical problem visible'])}",
                'LOW': f"Picture shows minor {issue} with {random.choice(['slight electrical issue', 'cosmetic damage to fixtures', 'minor malfunction'])}"
            },
            'garbage': {
                'HIGH': f"Image shows {issue} with {random.choice(['overflowing waste everywhere', 'health hazard visible', 'pest infestation', 'strong odor implied'])}",
                'MEDIUM': f"Photo of {issue} showing {random.choice(['significant waste accumulation', 'full garbage bins', 'scattered litter'])}",
                'LOW': f"Picture shows minor {issue} with {random.choice(['small amount of litter', 'minor waste issue', 'cosmetic cleanliness problem'])}"
            },
            'water_supply': {
                'HIGH': f"Image shows {issue} with {random.choice(['major water leakage', 'contaminated water visible', 'severe supply disruption', 'health risk evident'])}",
                'MEDIUM': f"Photo of {issue} showing {random.choice(['water leakage', 'supply problems', 'plumbing issues visible'])}",
                'LOW': f"Picture shows minor {issue} with {random.choice(['slight leakage', 'minor water issue', 'small plumbing problem'])}"
            },
            'public_safety': {
                'HIGH': f"Image shows {issue} with {random.choice(['serious structural damage', 'immediate danger visible', 'safety barriers needed', 'hazardous conditions clear'])}",
                'MEDIUM': f"Photo of {issue} showing {random.choice(['moderate safety concerns', 'structural problems', 'maintenance needed'])}",
                'LOW': f"Picture shows minor {issue} with {random.choice(['slight structural issue', 'cosmetic safety problem', 'minor maintenance needed'])}"
            }
        }
        
        base_desc = base_descriptions[category][urgency]
        
        # Add environmental context
        time_context = random.choice(['during daytime', 'in the evening', 'morning shot', 'afternoon lighting'])
        weather_context = random.choice(['clear weather', 'after rain', 'sunny conditions', 'overcast sky'])
        
        return f"{base_desc} {location}. Photo taken {time_context} with {weather_context}. {random.choice(['Good image quality', 'Clear visibility', 'Multiple angles shown', 'Close-up detail visible'])}."

    def add_contextual_details(self, text, category, urgency):
        """Add realistic contextual details to descriptions"""
        details = []
        
        # Add time context
        if urgency == 'HIGH':
            time_details = ['This happened yesterday', 'Noticed this morning', 'Just saw this', 'Happening right now']
        elif urgency == 'MEDIUM':  
            time_details = ['Been like this for a few days', 'Ongoing issue', 'Getting worse', 'Noticed last week']
        else:
            time_details = ['Been there for a while', 'Old issue', 'Gradually developing', 'Long-standing problem']
        
        details.append(random.choice(time_details))
        
        # Add impact details
        if category in ['road_problems', 'drainage']:
            impact_details = ['affecting traffic', 'causing delays', 'people complaining', 'getting media attention']
            if urgency == 'HIGH':
                details.append(f"Already {random.choice(['caused accidents', 'injured someone', 'damaged vehicles'])}")
            elif urgency == 'MEDIUM':
                details.append(f"Starting to {random.choice(impact_details)}")
        
        # Add location-specific details
        location_detail = random.choice(['Near the main entrance', 'Close to residential homes', 'Visible from the road', 'In a busy area'])
        details.append(location_detail)
        
        return f" {'. '.join(details)}."

    def generate_sample(self, category, urgency):
        """Generate a single training sample"""
        category_data = self.categories[category]
        issue = random.choice(category_data['issues'])
        location = random.choice(category_data['locations'])
        
        # Generate text description based on urgency
        if urgency == 'HIGH':
            text_desc = self.generate_high_urgency_description(category, issue, location)
        elif urgency == 'MEDIUM':
            text_desc = self.generate_medium_urgency_description(category, issue, location)
        else:
            text_desc = self.generate_low_urgency_description(category, issue, location)
        
        # Add contextual details
        text_desc += self.add_contextual_details(text_desc, category, urgency)
        
        # Add emotional tone variation
        tone = random.choice(list(self.emotional_tones.keys()))
        if random.random() < 0.3:  # 30% chance to add emotional prefix
            emotional_prefix = random.choice(self.emotional_tones[tone])
            text_desc = f"{emotional_prefix} {text_desc.lower()}"
        
        # Generate corresponding image description
        image_desc = self.generate_image_description(category, urgency, issue, location)
        
        # Generate urgency score within range
        score_range = self.urgency_levels[urgency]['score_range']
        urgency_score = random.randint(score_range[0], score_range[1])
        
        return {
            'id': f"{category}_{urgency}_{random.randint(1000, 9999)}",
            'category': category,
            'urgency_level': urgency,
            'urgency_score': urgency_score,
            'text_description': text_desc,
            'image_description': image_desc,
            'issue_type': issue,
            'location_type': location,
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 30))
        }

    def generate_dataset(self, samples_per_combination=500):
        """Generate complete dataset"""
        dataset = []
        total_combinations = len(self.categories) * len(self.urgency_levels)
        
        print(f"Generating dataset with {samples_per_combination} samples per combination...")
        print(f"Total combinations: {total_combinations}")
        print(f"Total samples to generate: {total_combinations * samples_per_combination}")
        
        for category in self.categories.keys():
            for urgency in self.urgency_levels.keys():
                print(f"Generating {samples_per_combination} samples for {category} - {urgency}...")
                
                for i in range(samples_per_combination):
                    sample = self.generate_sample(category, urgency)
                    dataset.append(sample)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{samples_per_combination} samples")
        
        return dataset

    def save_dataset(self, dataset, output_dir="../data"):
        """Save dataset in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "civic_issues_dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved as CSV: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(output_dir, "civic_issues_dataset.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, default=str, ensure_ascii=False)
        print(f"Dataset saved as JSON: {json_path}")
        
        # Generate summary statistics
        summary = {
            'total_samples': len(dataset),
            'categories': df['category'].value_counts().to_dict(),
            'urgency_levels': df['urgency_level'].value_counts().to_dict(),
            'urgency_score_stats': {
                'mean': df['urgency_score'].mean(),
                'median': df['urgency_score'].median(),
                'min': df['urgency_score'].min(),
                'max': df['urgency_score'].max()
            },
            'text_length_stats': {
                'mean': df['text_description'].str.len().mean(),
                'median': df['text_description'].str.len().median(),
                'min': df['text_description'].str.len().min(),
                'max': df['text_description'].str.len().max()
            }
        }
        
        summary_path = os.path.join(output_dir, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Dataset summary saved: {summary_path}")
        
        return df, summary

def main():
    """Main function to generate the dataset"""
    generator = CivicIssueDatasetGenerator()
    
    # Generate dataset with 500+ samples per category-urgency combination
    dataset = generator.generate_dataset(samples_per_combination=500)
    
    # Save dataset
    df, summary = generator.save_dataset(dataset)
    
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total samples generated: {len(dataset)}")
    print(f"Categories: {list(generator.categories.keys())}")
    print(f"Urgency levels: {list(generator.urgency_levels.keys())}")
    print(f"Samples per combination: 500")
    
    print("\nSample distribution:")
    print(df.groupby(['category', 'urgency_level']).size().unstack(fill_value=0))
    
    print(f"\nFiles saved in 'data' directory:")
    print("- civic_issues_dataset.csv")
    print("- civic_issues_dataset.json") 
    print("- dataset_summary.json")

if __name__ == "__main__":
    main()