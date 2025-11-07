---
title: Civic Issue Urgency Classifier
emoji: 🚨
colorFrom: gray
colorTo: gray
sdk: docker
pinned: true
license: mit
short_description: AI-powered text analysis for civic issue prioritization
---

# 🏛️ Civic Issue Urgency Classifier - Hugging Face Spaces

<div align="center">

![Spaces](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-blue)
![Python](https://img.shields.io/badge/python-3.12+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

**Text-Only NLP Classifier for Government Civic Issue Prioritization**

[Live Demo](#-live-demo) • [Quick Start](#-quick-start) • [API](#-api-usage) • [Model Info](#-model-information) • [Limitations](#-limitations)

</div>

---

## 📖 Overview

**Civic Issue Urgency Classifier** is an AI-powered text analysis system that helps governments prioritize citizen-reported civic issues. It analyzes text descriptions and automatically assigns urgency levels (HIGH/MEDIUM/LOW) with confidence scores and actionable recommendations.

**This is a text-only classifier.** Image analysis is not currently supported.

---

## 🎯 Quick Start

### Online (Hugging Face Spaces)
1. ✅ Open this Space (you're already here!)
2. 📝 Enter a civic issue description
3. 🎯 Click "Classify Urgency"
4. 📊 View priority level, score, and recommendations

### Local Development

**Prerequisites:**
- Python 3.12+
- pip or conda

**Installation:**
```bash
git clone https://github.com/nitish-niraj/urgency-checker
cd urgency-checker
pip install -r requirements.txt
python -m uvicorn src.demo_api_browser:app --reload --port 8001
```

**Then open:** http://localhost:8001

---

## 🎮 Live Demo

Try these example texts:

**HIGH Priority:**
```
Dangerous cracks on main road are blocking ambulance access during emergency. 
Multiple vehicles stuck, creating traffic hazard.
```

**MEDIUM Priority:**
```
Sidewalk near elementary school is uneven and cracked. 
Students using it daily, potential injury risk.
```

**LOW Priority:**
```
Park bench paint is fading and needs touch-up. 
Aesthetics need improvement for public spaces.
```

---

## 🔌 API Usage

### Base URL (Spaces)
```
https://huggingface.co/spaces/niru-nny/urgency-checker/api
```

### Endpoints

#### 1. **Health Check**
```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-07T10:30:00Z",
  "uptime": 3600
}
```

---

#### 2. **Classify Urgency** (Main Endpoint)
```bash
POST /classify-urgency
Content-Type: application/json

{
  "text": "Your civic issue description here"
}
```

**Request Requirements:**
- Text length: 10-5000 characters
- Language: English (recommended)
- Format: Plain text only

**Response:**
```json
{
  "priority_level": "HIGH",
  "urgency_score": 9.2,
  "confidence": 0.94,
  "recommended_department": "Road Safety",
  "breakdown": {
    "critical_keywords": [
      "dangerous",
      "blocking",
      "ambulance"
    ],
    "sentiment": "negative",
    "urgency_signals": 7
  },
  "processing_time_ms": 245
}
```

**Example with curl:**
```bash
curl -X POST https://huggingface.co/spaces/niru-nny/urgency-checker/api/classify-urgency \
  -H "Content-Type: application/json" \
  -d '{"text": "Dangerous cracks on road blocking ambulance access"}'
```

---

#### 3. **Statistics**
```bash
GET /stats
```

**Response:**
```json
{
  "total_classifications": 1523,
  "average_confidence": 0.87,
  "priority_distribution": {
    "HIGH": 34,
    "MEDIUM": 48,
    "LOW": 18
  },
  "average_response_time_ms": 234
}
```

---

## ⚙️ Model Information

### Architecture
- **Framework:** scikit-learn RandomForest (100 trees)
- **Feature Extraction:** TF-IDF vectorization
- **NLP Analysis:** TextBlob sentiment analysis
- **Input:** Text description (10-5000 characters)
- **Output:** Priority level, urgency score, confidence

### Performance
- **Average Response Time:** <3 seconds per request
- **Model Accuracy:** 87% on test set
- **Supported Languages:** English (primarily)
- **Concurrent Requests:** ~5-10 (CPU dependent)

### Model Details
```
Text Input
  ↓
Tokenization & Preprocessing
  ↓
TF-IDF Feature Extraction + TextBlob Sentiment
  ↓
RandomForest Classifier (100 trees ensemble)
  ↓
Urgency Score (1-10) + Priority Level (LOW/MEDIUM/HIGH)
  ↓
Confidence Score & Department Routing
  ↓
JSON Response
```

### Feature Set
1. **Lexical Features:**
   - TF-IDF weighted terms
   - Word frequency patterns
   - Text length statistics

2. **Semantic Features:**
   - TextBlob polarity (sentiment)
   - TextBlob subjectivity
   - Critical keyword detection

3. **Context Features:**
   - Department classification
   - Urgency signal patterns
   - Issue severity indicators

---

## ⚠️ Limitations

### What This System Can Do
✅ Analyze text-based civic issue descriptions  
✅ Assign priority levels (HIGH/MEDIUM/LOW)  
✅ Provide urgency scores with confidence  
✅ Recommend department routing  
✅ Process English text descriptions  

### What This System CANNOT Do
❌ Analyze images or attachments  
❌ Process non-English languages (not trained)  
❌ Replace human judgment for critical issues  
❌ Provide legal advice  
❌ Access real-time external data  
❌ Make final decisions (advisory only)  

### Important Disclaimers
- **Advisory Only:** Use as a decision support tool, not final authority
- **Language:** Optimized for English; other languages untested
- **Context:** Works best with detailed, descriptive text
- **Emergency:** Always prioritize safety for life-threatening issues
- **Bias:** Model reflects training data; may have inherent biases

---

## 📊 Example Results

### Case 1: High Priority
**Input:** "Dangerous gas leak detected near residential building. Strong smell reported by multiple residents. Evacuation may be needed."

**Output:**
```json
{
  "priority_level": "HIGH",
  "urgency_score": 9.8,
  "confidence": 0.96,
  "recommended_department": "Public Safety",
  "critical_keywords": ["dangerous", "gas", "leak", "evacuation"]
}
```

### Case 2: Medium Priority
**Input:** "Pothole in middle of busy street. About 1 foot deep. Car tires getting damaged."

**Output:**
```json
{
  "priority_level": "MEDIUM",
  "urgency_score": 6.5,
  "confidence": 0.91,
  "recommended_department": "Road Maintenance",
  "critical_keywords": ["pothole", "damaged"]
}
```

### Case 3: Low Priority
**Input:** "Park grass could use mowing. Looks a bit overgrown."

**Output:**
```json
{
  "priority_level": "LOW",
  "urgency_score": 2.1,
  "confidence": 0.88,
  "recommended_department": "Parks & Recreation",
  "critical_keywords": []
}
```

---

## 🚀 Deployment

### Hugging Face Spaces (This Deployment)
- **Type:** Docker container
- **Base Image:** Python 3.12-slim
- **Port:** 7860
- **Workers:** 2 (Uvicorn)
- **Health Check:** Every 30 seconds
- **Auto-scaling:** Managed by HF

### Local Deployment
```bash
# Development
python -m uvicorn src.demo_api_browser:app --reload --port 8001

# Production (local)
python -m uvicorn src.demo_api_browser:app --host 0.0.0.0 --port 8001 --workers 4

# Docker
docker build -t urgency-classifier .
docker run -p 7860:7860 urgency-classifier
```

### Environment Variables
```bash
PORT=7860              # Override default port
DEBUG=true             # Enable detailed error messages (optional)
```

---

## 🔐 Security & Privacy

- **Text Processing:** Text is processed in-memory, not stored
- **API Security:** CORS enabled for browser access
- **Input Validation:** 10-5000 character limit prevents abuse
- **Error Handling:** User-friendly errors, no stack traces exposed
- **Data Privacy:** No logging of user inputs or classifications

---

## 📝 API Error Handling

### Example Error Responses

**400 - Invalid Input (Too Short):**
```json
{
  "error": "Text too short",
  "message": "Please provide at least 10 characters",
  "min_length": 10,
  "current_length": 3
}
```

**400 - Invalid Input (Too Long):**
```json
{
  "error": "Text too long",
  "message": "Text cannot exceed 5000 characters",
  "max_length": 5000,
  "current_length": 6200
}
```

**500 - Server Error:**
```json
{
  "error": "Classification error",
  "message": "An unexpected error occurred during classification",
  "timestamp": "2025-11-07T10:30:00Z"
}
```

---

## 🆘 Troubleshooting

### Issue: "Connection Timeout"
- Space may be starting up. Wait 10-15 seconds and retry.
- Check HF Space status page.

### Issue: "Invalid Text Length"
- Ensure text is between 10-5000 characters.
- Remove extra whitespace or special characters.

### Issue: "Model Not Found"
- Model files should be in `/models/` directory.
- Verify Git LFS objects are properly downloaded.

### Issue: "High Response Time"
- CPU-intensive operation. May take 2-3 seconds.
- Try shorter text for faster processing.
- If consistently slow, model may need optimization.

---

## 📚 Documentation Files

- **README.md** (this file) - User documentation
- **DEPLOYMENT_HF.md** - HF Spaces deployment guide
- **DEPLOYMENT_READY.md** - Full deployment checklist
- **FINAL_VERIFICATION_CHECKLIST.md** - Pre-deployment verification

---

## 🤝 Contributing

Found a bug? Want to improve the model?

1. Fork: https://github.com/nitish-niraj/urgency-checker
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m "Add improvement"`
4. Push: `git push origin feature/improvement`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.

---

## 📧 Support

- **GitHub Issues:** https://github.com/nitish-niraj/urgency-checker/issues
- **HF Space:** This page (check "Community" tab)
- **Documentation:** See files listed above

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML Models: [scikit-learn](https://scikit-learn.org/)
- NLP: [TextBlob](https://textblob.readthedocs.io/)
- Hosting: [Hugging Face Spaces](https://huggingface.co/spaces)

---

<div align="center">

**⭐ If this project helps you, please consider giving it a star on GitHub! ⭐**

Made with ❤️ for civic engagement and government efficiency

</div>
