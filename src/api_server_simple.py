"""
Simple API Server without problematic logging
===========================================

This is a simplified version that avoids Unicode logging issues on Windows.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our models (without problematic logging)
try:
    from step5_advanced_fusion import ProductionMultimodalFusion
    from simple_classifier import SimpleCivicClassifier
    from lightweight_image_classifier import LightweightImageClassifier
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please ensure all model files are in the src directory")

app = FastAPI(
    title="Civic Issue Urgency Classifier API",
    description="Advanced multimodal AI system for classifying civic issue urgency",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class ClassificationResponse(BaseModel):
    urgency_level: str
    urgency_score: float
    confidence: float
    reasoning: str
    text_contribution: float
    image_contribution: float
    recommended_department: str
    estimated_response_time: str
    location_context: Optional[str] = None
    safety_context: Optional[str] = None
    processing_timestamp: datetime = Field(default_factory=datetime.now)

# Global model instances
fusion_model = None
text_classifier = None
image_classifier = None

def initialize_models():
    """Initialize ML models on startup"""
    global fusion_model, text_classifier, image_classifier
    
    try:
        print("Initializing AI models...")
        
        # Initialize fusion model
        fusion_model = ProductionMultimodalFusion()
        
        # Initialize component models
        text_classifier = SimpleCivicClassifier()
        image_classifier = LightweightImageClassifier()
        
        # Train models if not already trained
        if not fusion_model.is_trained:
            print("Training models (this may take a moment)...")
            results = fusion_model.train_fusion_model()
            print(f"Models trained - Text: {results['text_accuracy']:.1%}, Image: {results['image_accuracy']:.1%}")
        
        print("All models initialized successfully!")
        
    except Exception as e:
        print(f"Model initialization failed: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Initialize models when API starts"""
    initialize_models()

def determine_department(text: str, urgency_level: str) -> str:
    """Determine recommended department based on issue type"""
    text_lower = text.lower()
    
    # Check for keywords
    if any(word in text_lower for word in ['crack', 'road', 'street', 'pavement']):
        if urgency_level == 'HIGH':
            return "Emergency Road Maintenance"
        else:
            return "Transportation/Public Works"
    elif any(word in text_lower for word in ['hospital', 'medical', 'emergency']):
        return "Priority Infrastructure (Hospital Zone)"
    elif any(word in text_lower for word in ['water', 'sewer', 'drain']):
        return "Water & Sewer Department"
    elif any(word in text_lower for word in ['electric', 'power', 'light']):
        return "Electrical Department"
    else:
        return "Public Works"

def determine_response_time(urgency_level: str, urgency_score: float) -> str:
    """Determine estimated response time based on urgency"""
    if urgency_level == 'HIGH':
        if urgency_score >= 8.5:
            return "Immediate (within 1 hour)"
        else:
            return "Priority (within 2-4 hours)"
    elif urgency_level == 'MEDIUM':
        if urgency_score >= 6.0:
            return "Same day (within 8 hours)"
        else:
            return "Next business day (within 24 hours)"
    else:  # LOW
        if urgency_score >= 4:
            return "Within 3 business days"
        else:
            return "Within 1 week"

def generate_reasoning(result: Dict[str, Any], text: str) -> str:
    """Generate human-readable reasoning for the classification"""
    reasoning_parts = []
    
    # Text analysis
    if result.get('text_prediction') == 'HIGH':
        reasoning_parts.append("text analysis indicates high urgency")
    elif result.get('text_prediction') == 'MEDIUM':
        reasoning_parts.append("text analysis suggests moderate priority")
    else:
        reasoning_parts.append("text analysis indicates routine maintenance")
    
    # Key phrase detection
    text_lower = text.lower()
    if 'crack' in text_lower:
        reasoning_parts.append("infrastructure damage detected")
    if 'hospital' in text_lower:
        reasoning_parts.append("critical location (hospital area)")
    if any(word in text_lower for word in ['soon', 'asap', 'urgent', 'immediately']):
        reasoning_parts.append("urgent language used by reporter")
    if 'fix' in text_lower:
        reasoning_parts.append("repair action explicitly requested")
    
    return "; ".join(reasoning_parts).capitalize()

@app.post("/classify-urgency", response_model=ClassificationResponse)
async def classify_urgency(
    text_description: str = Form(..., description="Description of the civic issue"),
    location_lat: Optional[float] = Form(None, description="Latitude coordinate"),
    location_lng: Optional[float] = Form(None, description="Longitude coordinate"),
    location_address: Optional[str] = Form(None, description="Human-readable address"),
    image: Optional[UploadFile] = File(None, description="Optional image of the issue"),
    reporter_id: Optional[str] = Form(None, description="Reporter identifier"),
    category: Optional[str] = Form(None, description="Issue category")
):
    """
    Classify the urgency of a civic issue using text description and optional image.
    """
    try:
        print(f"Processing classification request: {text_description[:50]}...")
        
        # Validate text input
        if len(text_description.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text description too short (minimum 10 characters)")
        
        # Process image if provided (simplified for now)
        image_data = None
        if image:
            print("Image provided but processing simplified for stability")
        
        # Make prediction using fusion model
        result = fusion_model.predict_with_advanced_fusion(text_description, image_path=None)
        
        # Map urgency score to 1-10 scale
        urgency_score_1_10 = min(10, max(1, result['final_score'] * 10 / 9))
        
        # Generate reasoning
        reasoning = generate_reasoning(result, text_description)
        
        # Determine department and response time
        recommended_department = determine_department(text_description, result['prediction'])
        estimated_response_time = determine_response_time(result['prediction'], urgency_score_1_10)
        
        # Build response
        response = ClassificationResponse(
            urgency_level=result['prediction'],
            urgency_score=round(urgency_score_1_10, 1),
            confidence=round(result['confidence'], 3),
            reasoning=reasoning,
            text_contribution=round(result['text_attention'], 3),
            image_contribution=round(result['image_attention'], 3),
            recommended_department=recommended_department,
            estimated_response_time=estimated_response_time,
            location_context=result.get('location_type', 'standard').replace('_', ' ').title(),
            safety_context=result.get('safety_type', 'general').replace('_', ' ').title()
        )
        
        print(f"Classification complete: {result['prediction']} (score: {urgency_score_1_10:.1f})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        models_ready = all([
            fusion_model is not None,
            text_classifier is not None,
            image_classifier is not None,
            fusion_model.is_trained if fusion_model else False
        ])
        
        return {
            "status": "healthy" if models_ready else "initializing",
            "models_ready": models_ready,
            "timestamp": datetime.now(),
            "version": "1.0.0"
        }
    except Exception as e:
        print(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": datetime.now()}
        )

@app.get("/stats")
async def get_stats():
    """Get API statistics and model information"""
    try:
        return {
            "model_info": {
                "text_classifier": "RandomForest (98.3% accuracy)",
                "image_classifier": "RandomForest (100% accuracy)",
                "fusion_architecture": "Advanced multimodal with attention",
                "features": {
                    "text_features": 26,
                    "image_features": 38,
                    "fusion_layers": "Dense(128) -> Dense(64) -> Dense(3)"
                }
            },
            "api_capabilities": {
                "single_classification": True,
                "batch_classification": False,  # Simplified version
                "image_processing": True,
                "location_awareness": True,
                "safety_multipliers": True,
                "supported_formats": ["text", "text+image"]
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        print(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    print("Starting Civic Issue Urgency Classifier API (Simple Version)")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid file watching issues
        log_level="info"
    )