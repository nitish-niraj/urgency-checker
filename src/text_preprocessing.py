"""
Civic Issue Urgency Classifier - Text Preprocessing Pipeline
Comprehensive text processing for multimodal AI classification system
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import pickle
import json
from datetime import datetime

# Natural Language Processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class CivicIssueTextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessing pipeline"""
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define keyword dictionaries for feature extraction
        self.location_indicators = {
            'institutional': [
                'hospital', 'school', 'college', 'university', 'clinic', 'dispensary',
                'government office', 'municipal office', 'police station', 'fire station',
                'court', 'library', 'post office', 'bank', 'atm'
            ],
            'transportation': [
                'main road', 'highway', 'street', 'lane', 'avenue', 'bridge', 'flyover',
                'bus stop', 'railway station', 'metro station', 'airport', 'parking',
                'intersection', 'crossing', 'traffic signal', 'roundabout'
            ],
            'residential': [
                'residential', 'colony', 'society', 'apartment', 'flat', 'house',
                'neighborhood', 'locality', 'area', 'sector', 'block', 'plot'
            ],
            'commercial': [
                'market', 'mall', 'shop', 'store', 'restaurant', 'hotel', 'office',
                'commercial', 'business', 'plaza', 'complex', 'center', 'bazaar'
            ],
            'public': [
                'park', 'garden', 'playground', 'stadium', 'community center',
                'public toilet', 'bus stand', 'taxi stand', 'public area'
            ]
        }
        
        self.safety_keywords = {
            'high_danger': [
                'danger', 'dangerous', 'hazard', 'hazardous', 'emergency', 'urgent',
                'critical', 'serious', 'life threatening', 'accident', 'injury',
                'death', 'fatal', 'electrocution', 'fire', 'explosion', 'collapse'
            ],
            'structural_issues': [
                'broken', 'damaged', 'cracked', 'exposed', 'leaking', 'burst',
                'fallen', 'loose', 'unstable', 'sharp', 'protruding'
            ],
            'health_hazards': [
                'contaminated', 'polluted', 'toxic', 'sewage', 'waste', 'garbage',
                'smell', 'stench', 'disease', 'infection', 'pest', 'mosquito'
            ]
        }
        
        self.severity_descriptors = {
            'high_severity': [
                'completely', 'totally', 'entirely', 'fully', 'severely', 'badly',
                'deep', 'large', 'huge', 'massive', 'major', 'extensive'
            ],
            'medium_severity': [
                'partially', 'somewhat', 'moderately', 'fairly', 'quite',
                'medium', 'average', 'normal', 'regular'
            ],
            'low_severity': [
                'slightly', 'minor', 'small', 'little', 'tiny', 'minimal',
                'cosmetic', 'surface', 'superficial'
            ]
        }
        
        self.impact_indicators = {
            'traffic_impact': [
                'traffic jam', 'traffic block', 'road block', 'congestion',
                'vehicles stuck', 'cannot pass', 'detour', 'alternate route'
            ],
            'people_affected': [
                'people affected', 'residents suffering', 'citizens complaining',
                'families affected', 'children at risk', 'elderly problems'
            ],
            'service_disruption': [
                'no water', 'power cut', 'electricity gone', 'service stopped',
                'supply disrupted', 'not working', 'out of order'
            ]
        }
        
        self.time_indicators = {
            'immediate': [
                'now', 'immediately', 'right now', 'urgent', 'asap',
                'today', 'this moment', 'currently'
            ],
            'recent': [
                'yesterday', 'last night', 'this morning', 'few hours ago',
                'since morning', 'since evening', 'today'
            ],
            'ongoing': [
                'for days', 'for weeks', 'for months', 'since long',
                'continuing', 'still happening', 'ongoing', 'persistent'
            ]
        }
        
        # Abbreviations and slang dictionary
        self.abbreviations = {
            'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
            'govt': 'government', 'mcd': 'municipal corporation', 'nagar': 'city',
            'chowk': 'square', 'gali': 'lane', 'marg': 'road', 'vihar': 'colony',
            'nagar': 'city', 'puram': 'town', 'abad': 'city', 'garh': 'fort city'
        }
        
        # Common civic issue slang
        self.slang_replacements = {
            'pothole': 'pothole', 'pot hole': 'pothole',
            'waterlogging': 'water stagnation', 'water logging': 'water stagnation',
            'streetlight': 'street light', 'street lamp': 'street light',
            'manholes': 'manhole', 'sewer hole': 'manhole'
        }
        
        # Coordinate patterns (latitude, longitude)
        self.coordinate_pattern = re.compile(
            r'(?:lat|latitude)[:\s]*([+-]?\d+\.?\d*)[,\s]*(?:lon|lng|longitude)[:\s]*([+-]?\d+\.?\d*)',
            re.IGNORECASE
        )
        
        # Initialize TF-IDF vectorizers
        self.tfidf_vectorizer = None
        self.safety_vectorizer = None
        self.location_vectorizer = None
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        
    def download_required_packages(self):
        """Download and install required packages"""
        try:
            import textblob
        except ImportError:
            print("Installing TextBlob...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'textblob'])
            import textblob
    
    def clean_text(self, text):
        """Basic text cleaning and normalization"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common abbreviations
        for abbrev, full_form in self.abbreviations.items():
            text = re.sub(r'\b' + abbrev + r'\b', full_form, text)
        
        # Handle slang replacements
        for slang, replacement in self.slang_replacements.items():
            text = text.replace(slang, replacement)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove numbers that are not coordinates or relevant measurements
        text = re.sub(r'\b\d{4,}\b', '', text)  # Remove long numbers like phone numbers
        
        return text.strip()
    
    def extract_coordinates(self, text):
        """Extract geographical coordinates from text"""
        coordinates = []
        matches = self.coordinate_pattern.findall(text)
        
        for match in matches:
            try:
                lat, lon = float(match[0]), float(match[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coordinates.append({'latitude': lat, 'longitude': lon})
            except ValueError:
                continue
        
        return coordinates
    
    def extract_location_features(self, text):
        """Extract location-related features"""
        features = {
            'institutional_locations': 0,
            'transportation_locations': 0,
            'residential_locations': 0,
            'commercial_locations': 0,
            'public_locations': 0,
            'location_mentions': []
        }
        
        for location_type, keywords in self.location_indicators.items():
            count = 0
            mentions = []
            for keyword in keywords:
                if keyword in text:
                    count += text.count(keyword)
                    mentions.append(keyword)
            
            features[f'{location_type}_locations'] = count
            if mentions:
                features['location_mentions'].extend(mentions)
        
        return features
    
    def extract_safety_features(self, text):
        """Extract safety-related keywords and features"""
        features = {
            'high_danger_keywords': 0,
            'structural_issue_keywords': 0,
            'health_hazard_keywords': 0,
            'safety_score': 0,
            'safety_mentions': []
        }
        
        total_safety_keywords = 0
        
        for safety_type, keywords in self.safety_keywords.items():
            count = 0
            mentions = []
            for keyword in keywords:
                if keyword in text:
                    keyword_count = text.count(keyword)
                    count += keyword_count
                    total_safety_keywords += keyword_count
                    mentions.append(keyword)
            
            features[f'{safety_type}_keywords'] = count
            if mentions:
                features['safety_mentions'].extend(mentions)
        
        # Calculate safety score (0-1)
        features['safety_score'] = min(total_safety_keywords / 5.0, 1.0)
        
        return features
    
    def extract_severity_features(self, text):
        """Extract severity descriptors"""
        features = {
            'high_severity_words': 0,
            'medium_severity_words': 0,
            'low_severity_words': 0,
            'severity_score': 0.5,  # Default medium
            'severity_mentions': []
        }
        
        severity_scores = {'high_severity': 1.0, 'medium_severity': 0.5, 'low_severity': 0.1}
        weighted_score = 0
        total_words = 0
        
        for severity_level, words in self.severity_descriptors.items():
            count = 0
            mentions = []
            for word in words:
                if word in text:
                    word_count = text.count(word)
                    count += word_count
                    total_words += word_count
                    weighted_score += word_count * severity_scores[severity_level]
                    mentions.append(word)
            
            features[f'{severity_level}_words'] = count
            if mentions:
                features['severity_mentions'].extend(mentions)
        
        if total_words > 0:
            features['severity_score'] = weighted_score / total_words
        
        return features
    
    def extract_impact_features(self, text):
        """Extract impact indicators"""
        features = {
            'traffic_impact_keywords': 0,
            'people_affected_keywords': 0,
            'service_disruption_keywords': 0,
            'impact_score': 0,
            'impact_mentions': []
        }
        
        total_impact = 0
        
        for impact_type, keywords in self.impact_indicators.items():
            count = 0
            mentions = []
            for keyword in keywords:
                if keyword in text:
                    keyword_count = text.count(keyword)
                    count += keyword_count
                    total_impact += keyword_count
                    mentions.append(keyword)
            
            features[f'{impact_type}_keywords'] = count
            if mentions:
                features['impact_mentions'].extend(mentions)
        
        features['impact_score'] = min(total_impact / 3.0, 1.0)
        
        return features
    
    def extract_time_features(self, text):
        """Extract time-related urgency indicators"""
        features = {
            'immediate_time_keywords': 0,
            'recent_time_keywords': 0,
            'ongoing_time_keywords': 0,
            'urgency_time_score': 0.5,
            'time_mentions': []
        }
        
        time_scores = {'immediate': 1.0, 'recent': 0.7, 'ongoing': 0.8}
        weighted_score = 0
        total_keywords = 0
        
        for time_type, keywords in self.time_indicators.items():
            count = 0
            mentions = []
            for keyword in keywords:
                if keyword in text:
                    keyword_count = text.count(keyword)
                    count += keyword_count
                    total_keywords += keyword_count
                    weighted_score += keyword_count * time_scores[time_type]
                    mentions.append(keyword)
            
            features[f'{time_type}_time_keywords'] = count
            if mentions:
                features['time_mentions'].extend(mentions)
        
        if total_keywords > 0:
            features['urgency_time_score'] = weighted_score / total_keywords
        
        return features
    
    def perform_sentiment_analysis(self, text):
        """Perform sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            return {
                'sentiment_polarity': blob.sentiment.polarity,  # -1 to 1
                'sentiment_subjectivity': blob.sentiment.subjectivity,  # 0 to 1
                'sentiment_urgency_score': abs(blob.sentiment.polarity)  # Absolute urgency
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.5,
                'sentiment_urgency_score': 0.0
            }
    
    def extract_named_entities(self, text):
        """Extract named entities using NLTK"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = {
                'person_entities': [],
                'location_entities': [],
                'organization_entities': [],
                'entity_count': 0
            }
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    entity_label = chunk.label()
                    
                    if entity_label == 'PERSON':
                        entities['person_entities'].append(entity_text)
                    elif entity_label in ['GPE', 'LOCATION']:  # Geopolitical entity, Location
                        entities['location_entities'].append(entity_text)
                    elif entity_label == 'ORGANIZATION':
                        entities['organization_entities'].append(entity_text)
                    
                    entities['entity_count'] += 1
            
            return entities
        
        except Exception as e:
            print(f"Named entity recognition error: {e}")
            return {
                'person_entities': [],
                'location_entities': [],
                'organization_entities': [],
                'entity_count': 0
            }
    
    def create_tfidf_features(self, texts, max_features=1000):
        """Create TF-IDF features for the text corpus"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix.toarray()
    
    def process_single_text(self, text):
        """Process a single text and extract all features"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Extract all feature sets
        features = {}
        
        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(cleaned_text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in cleaned_text.split()]) if cleaned_text.split() else 0
        
        # Extract coordinates
        coordinates = self.extract_coordinates(text)
        features['has_coordinates'] = len(coordinates) > 0
        features['coordinate_count'] = len(coordinates)
        features['coordinates'] = coordinates
        
        # Extract categorical features
        location_features = self.extract_location_features(cleaned_text)
        safety_features = self.extract_safety_features(cleaned_text)
        severity_features = self.extract_severity_features(cleaned_text)
        impact_features = self.extract_impact_features(cleaned_text)
        time_features = self.extract_time_features(cleaned_text)
        
        # Sentiment analysis
        sentiment_features = self.perform_sentiment_analysis(cleaned_text)
        
        # Named entity recognition
        entity_features = self.extract_named_entities(cleaned_text)
        
        # Combine all features
        features.update(location_features)
        features.update(safety_features)
        features.update(severity_features)
        features.update(impact_features)
        features.update(time_features)
        features.update(sentiment_features)
        features.update(entity_features)
        
        # Calculate composite urgency score
        features['composite_urgency_score'] = self.calculate_composite_urgency(features)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'features': features
        }
    
    def calculate_composite_urgency(self, features):
        """Calculate a composite urgency score from all features"""
        urgency_components = [
            features.get('safety_score', 0) * 0.3,
            features.get('severity_score', 0.5) * 0.25,
            features.get('impact_score', 0) * 0.2,
            features.get('urgency_time_score', 0.5) * 0.15,
            features.get('sentiment_urgency_score', 0) * 0.1
        ]
        
        return min(sum(urgency_components), 1.0)
    
    def process_dataset(self, texts):
        """Process entire dataset and return feature matrix"""
        print(f"Processing {len(texts)} texts...")
        
        processed_data = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processed {i}/{len(texts)} texts...")
            
            processed_item = self.process_single_text(text)
            processed_data.append(processed_item)
        
        # Create TF-IDF features for all texts
        cleaned_texts = [item['cleaned_text'] for item in processed_data]
        tfidf_features = self.create_tfidf_features(cleaned_texts)
        
        # Add TF-IDF features to each processed item
        for i, item in enumerate(processed_data):
            item['tfidf_features'] = tfidf_features[i]
        
        print("Text processing completed!")
        return processed_data
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor state"""
        preprocessor_state = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_scaler': self.feature_scaler,
            'location_indicators': self.location_indicators,
            'safety_keywords': self.safety_keywords,
            'severity_descriptors': self.severity_descriptors,
            'impact_indicators': self.impact_indicators,
            'time_indicators': self.time_indicators
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load the preprocessor state"""
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.tfidf_vectorizer = preprocessor_state['tfidf_vectorizer']
        self.feature_scaler = preprocessor_state['feature_scaler']
        
        print(f"Preprocessor loaded from {filepath}")

def main():
    """Main function to demonstrate text preprocessing"""
    # Initialize preprocessor
    preprocessor = CivicIssueTextPreprocessor()
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("../data/civic_issues_dataset.csv")
    
    # Sample a subset for demonstration (process first 100 samples)
    sample_texts = df['text_description'].head(100).tolist()
    
    print("\nDemonstrating text preprocessing on sample data...")
    
    # Process sample texts
    processed_data = preprocessor.process_dataset(sample_texts)
    
    # Display results for first few samples
    print("\n" + "="*80)
    print("SAMPLE PREPROCESSING RESULTS")
    print("="*80)
    
    for i in range(min(3, len(processed_data))):
        item = processed_data[i]
        features = item['features']
        
        print(f"\n🔍 SAMPLE {i+1}:")
        print(f"Original: {item['original_text'][:100]}...")
        print(f"Cleaned:  {item['cleaned_text'][:100]}...")
        
        print(f"\n📊 Key Features:")
        print(f"  Text Length: {features['text_length']}")
        print(f"  Word Count: {features['word_count']}")
        print(f"  Safety Score: {features['safety_score']:.3f}")
        print(f"  Severity Score: {features['severity_score']:.3f}")
        print(f"  Impact Score: {features['impact_score']:.3f}")
        print(f"  Time Urgency Score: {features['urgency_time_score']:.3f}")
        print(f"  Composite Urgency: {features['composite_urgency_score']:.3f}")
        print(f"  Sentiment Polarity: {features['sentiment_polarity']:.3f}")
        
        if features['safety_mentions']:
            print(f"  Safety Keywords: {features['safety_mentions'][:3]}")
        if features['location_mentions']:
            print(f"  Location Keywords: {features['location_mentions'][:3]}")
    
    print(f"\n✅ Preprocessing pipeline demonstration completed!")
    print(f"Processed {len(processed_data)} samples successfully.")

if __name__ == "__main__":
    main()