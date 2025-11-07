# 🏛️ Civic Issue Urgency Classifier

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

**AI-Powered Text Analysis System for Government Civic Issue Prioritization**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API Docs](#-api-documentation) • [Architecture](#-architecture)

</div>

---

## 📖 Overview

The **Civic Issue Urgency Classifier** is an AI-powered text analysis system that analyzes citizen-submitted civic issue descriptions and automatically assigns urgency scores (Low/Medium/High) for efficient government response prioritization.

### 🎯 Problem Statement

Government agencies receive thousands of civic issue reports daily. Manual triage is:
- ⏰ **Time-consuming** - Hours wasted sorting reports
- ❌ **Inconsistent** - Different staff = different priorities  
- 🚨 **Risky** - Critical issues may be delayed
- 💰 **Expensive** - Requires dedicated staff

### ✨ Our Solution

Automated AI-powered text classification system that:
- 🤖 **Analyzes** text descriptions using NLP sentiment analysis and TF-IDF
- � **Scores** urgency levels with confidence metrics
- 🏢 **Routes** issues to appropriate departments automatically
- ⚡ **Responds** instantly with actionable recommendations

> **Note:** Image analysis is currently unavailable. The system performs text-only classification.

---

## 🌟 Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **🔤 Text Analysis** | NLP-based sentiment analysis + TF-IDF vectorization | ✅ Active |
| **🤖 AI Classification** | RandomForest model for urgency prediction | ✅ Active |
| **🎯 Smart Routing** | Automatic department assignment | ✅ Active |
| **⏱️ Response Estimation** | AI-predicted resolution time | ✅ Active |
| **🖼️ Image Analysis** | Computer vision for damage detection | ⚠️ Coming Soon |

### Technical Features

- ✅ **Production-Ready API** - FastAPI with async support
- ✅ **Real-time Processing** - 2-3 second response time
- ✅ **Batch Processing** - Handle multiple issues simultaneously
- ✅ **Health Monitoring** - Built-in health checks and stats
- ✅ **Interactive UI** - Modern iOS 26-inspired liquid design
- ✅ **Comprehensive Logging** - Full audit trail
- ✅ **Scalable Architecture** - Cloud-ready deployment

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
Git
```

### Installation

**1️⃣ Clone the Repository**
```bash
git clone https://github.com/nitish-niraj/urgency-checker.git
cd urgency-checker
```

**2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Start the Application**
```bash
python start_ui.py
```

**4️⃣ Open Your Browser**
```
http://localhost:8001
```

That's it! 🎉 The system is now running!

---

## 💻 Usage

### Web Interface (Recommended)

1. **Navigate to** http://localhost:8001
2. **Enter** your civic issue description
3. **Add** location details (optional)
4. **Click** "Classify Urgency"
5. **View** instant AI analysis with recommendations

### API Usage

#### Example 1: Single Classification

```python
import requests

# Prepare civic issue data
data = {
    "text_description": "Dangerous cracks in road near hospital. Fix ASAP!",
    "location_address": "Near University Hospital",
    "category": "Infrastructure"
}

# Send classification request
response = requests.post(
    "http://localhost:8001/classify-urgency",
    json=data
)

result = response.json()
print(f"Urgency: {result['urgency_level']}")  # HIGH
print(f"Score: {result['urgency_score']}/10")  # 8.5/10
print(f"Department: {result['recommended_department']}")  # Emergency Services
```

#### Example 2: Batch Processing

```python
import requests

# Multiple civic issues
issues = [
    {"text_description": "Fire hazard in building!", "location_address": "Downtown"},
    {"text_description": "Minor graffiti on bench", "location_address": "City Park"}
]

# Batch classification
response = requests.post(
    "http://localhost:8001/batch-classify",
    json={"issues": issues}
)

results = response.json()
for i, result in enumerate(results['results']):
    print(f"Issue {i+1}: {result['urgency_level']}")
```

#### Example 3: System Health Check

```python
import requests

response = requests.get("http://localhost:8001/health")
health = response.json()

print(f"Status: {health['status']}")  # healthy
print(f"Version: {health['version']}")  # 1.0.0
```

---

## 📊 API Documentation

### Base URL
```
http://localhost:8001
```

### Endpoints

#### 🏠 Home Page
```http
GET /
```
Returns interactive web interface.

#### 🔍 Classify Civic Issue
```http
POST /classify-urgency
Content-Type: application/json

{
  "text_description": "Issue description",
  "location_address": "Location (optional)",
  "category": "Category (optional)"
}
```

**Response:**
```json
{
  "urgency_level": "HIGH",
  "urgency_score": 8.5,
  "confidence": 0.92,
  "recommended_department": "Emergency Services",
  "estimated_response_time": "Within 1 hour",
  "reasoning": "AI detected high urgency based on keywords: 'dangerous', 'cracks', 'hospital', 'ambulance'. Immediate action required for public safety.",
  "location_context": "Hospital",
  "safety_context": "Emergency"
}
```

#### 📊 System Statistics
```http
GET /stats
```

**Response:**
```json
{
  "service_name": "Civic Issue Urgency Classifier",
  "status": "operational",
  "model_info": {
    "text_classifier": "TextBlob + TF-IDF",
    "ai_model": "RandomForest Ensemble",
    "analysis_type": "Text-based NLP"
  },
  "performance_metrics": {
    "avg_response_time": "< 3 seconds",
    "total_requests": 247,
    "status": "Active"
  }
}
```

#### 💚 Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-10-14T10:30:00"
}
```

#### 🧪 Demo Classification
```http
GET /demo
```
Returns sample classification for testing.

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface (UI)                      │
│              iOS 26-inspired Liquid Design                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI REST API                          │
│            (Request Handling & Routing)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI Classification Engine                   │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Text Analysis│  │ AI Classifier│                        │
│  │  (NLP + TF- │  │ (RandomForest│                        │
│  │     IDF)     │  │   Ensemble)  │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Classification Output                        │
│  • Urgency Level (HIGH/MEDIUM/LOW)                         │
│  • Department Routing                                       │
│  • Response Time Estimation                                 │
│  • Action Recommendations                                   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- 🐍 Python 3.8+
- ⚡ FastAPI (Modern async web framework)
- 🤖 scikit-learn (Machine Learning)
- 📝 TextBlob (NLP & Sentiment Analysis)
- 🔢 NumPy & Pandas (Data Processing)

**Frontend:**
- 🎨 HTML5 + CSS3 (Liquid Design)
- ⚡ Vanilla JavaScript (No frameworks needed)
- 🎭 Glassmorphism UI
- 📱 Responsive Design

**AI Models:**
- 📊 RandomForest Classifier (Ensemble)
- 📝 TF-IDF Vectorizer (Text Features)
- � TextBlob (Sentiment Analysis)
- 🎯 Custom Feature Engineering

---

## 📁 Project Structure

```
urgency-checker/
├── 📄 README.md                        # This file
├── 📄 requirements.txt                 # Python dependencies
├── 📄 start_ui.py                      # Easy startup script
│
├── 📁 src/                             # Source code
│   ├── demo_api_browser.py             # Main API server
│   ├── step5_advanced_fusion.py        # AI fusion model
│   ├── text_preprocessing.py           # Text analysis
│   ├── integrate_fire_dataset.py       # Fire dataset integration
│   └── update_enhanced_metadata.py     # Dataset management
│
├── 📁 static/                          # UI assets
│   ├── 📁 css/
│   │   ├── styles.css                  # Main liquid design CSS
│   │   └── animations.css              # Animation effects
│   ├── 📁 js/
│   │   ├── main.js                     # Main interactions
│   │   └── animations.js               # UI animations
│   └── 📁 images/                      # Icons & backgrounds
│
├── 📁 templates/                       # HTML templates
│   ├── index.html                      # Landing page
│   ├── classify.html                   # Classification UI
│   └── dashboard.html                  # Stats dashboard
│
├── 📁 data/                            # Training data
│   ├── 📁 images_enhanced/             # Image dataset
│   │   ├── HIGH/ (30 fire images)
│   │   ├── MEDIUM/ (30 images)
│   │   └── LOW/ (30 images)
│   └── civic_issues.csv                # Text dataset
│
├── 📁 models/                          # Trained models
│   ├── text_classifier.pkl
│   ├── image_classifier.pkl
│   └── fusion_model.pkl
│
├── 📁 logs/                            # Application logs
│   └── api.log
│
└── 📁 docs/                            # Documentation
    ├── AI_MODELS_EXPLANATION.md
    ├── TEXT_SENTIMENT_ANALYSIS.md
    └── IMAGE_SAMPLES_EXPLANATION.md
```

---

## 🎨 UI Design Philosophy

Our interface follows **Apple's iOS 26 Liquid Design** principles:

### Design Elements

| Element | Description |
|---------|-------------|
| **Glassmorphism** | Frosted glass effects with backdrop blur |
| **Fluid Animations** | Smooth 60fps transitions |
| **Gradient Backgrounds** | Dynamic multi-color gradients |
| **Soft Shadows** | Elevated UI components |
| **Rounded Corners** | Organic, friendly shapes |
| **Interactive Feedback** | Hover, focus, and click animations |
| **Responsive** | Mobile, tablet, and desktop optimized |

---

## 📈 Performance Metrics

### Classification Model

| Component | Status | Description |
|-----------|--------|-------------|
| Text Classifier | ✅ Active | NLP-based urgency analysis |
| AI Ensemble | ✅ Active | RandomForest classification |
| Image Analysis | ⚠️ Coming Soon | Computer vision integration planned |

### System Performance

- ⚡ **Response Time:** Fast (< 3 seconds)
- 🔄 **Throughput:** Multiple concurrent requests
- 💾 **Memory Usage:** Lightweight (~200MB)
- 📊 **Analysis:** Text-only classification

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Test API Endpoint
```bash
python final_api_test.py
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py
```

---

## 🚀 Deployment

### Local Development
```bash
python start_ui.py
```

### Production (Docker)
```bash
docker build -t civic-classifier .
docker run -p 8001:8001 civic-classifier
```

### Cloud Deployment (AWS/Azure/GCP)
See `docs/DEPLOYMENT.md` for detailed cloud deployment guides.

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- ✅ Follow PEP 8 style guide
- ✅ Add unit tests for new features
- ✅ Update documentation
- ✅ Ensure all tests pass

---

## 📝 Use Cases

### Government Agencies
- 🏛️ Municipal governments
- 🚓 Public safety departments
- 🏗️ Infrastructure maintenance
- 🌳 Parks & recreation

### Smart Cities
- 📱 Citizen reporting apps
- 🗺️ Urban planning systems
- 🚦 Traffic management
- 🌐 IoT integration

### Private Sector
- 🏢 Property management
- 🏨 Facility management
- 🚗 Fleet operations
- 📞 Customer service

---

## 🎓 Research & References

This project implements techniques from:

- **NLP:** TextBlob sentiment analysis, TF-IDF vectorization
- **Machine Learning:** RandomForest ensemble classification
- **UI/UX:** Apple Human Interface Guidelines, iOS 26 design system
- **API Design:** RESTful architecture with FastAPI

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Nitish Niraj**
- GitHub: [@nitish-niraj](https://github.com/nitish-niraj)
- Repository: [urgency-checker](https://github.com/nitish-niraj/urgency-checker)

---

## 🙏 Acknowledgments

- **TextBlob** - Natural language processing library
- **FastAPI** - Modern web framework for building APIs
- **FastAPI** - Modern Python web framework
- **scikit-learn** - Machine learning library
- **Apple Design Team** - iOS 26 design inspiration

---

## 📞 Support

Need help? Have questions?

- 📧 **Email:** nitish.niraj@example.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/nitish-niraj/urgency-checker/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/nitish-niraj/urgency-checker/discussions)

---

## 🗺️ Roadmap

### Version 1.1 (Coming Soon)
- [ ] Mobile app (iOS & Android)
- [ ] Real-time notifications
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Integration with 311 systems

### Version 2.0 (Future)
- [ ] Deep learning models (YOLO, ResNet)
- [ ] Video analysis support
- [ ] Predictive maintenance
- [ ] Blockchain audit trail
- [ ] GraphQL API

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for smarter government services

[Report Bug](https://github.com/nitish-niraj/urgency-checker/issues) • [Request Feature](https://github.com/nitish-niraj/urgency-checker/issues)

</div>
