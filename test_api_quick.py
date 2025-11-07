"""
Quick Test Script for Civic Issue Classifier
============================================
Tests the API endpoint to ensure it's working
"""

import requests
import json

print("🧪 Testing Civic Issue Urgency Classifier API")
print("=" * 50)

# Test data
test_data = {
    "text_description": "There are dangerous cracks in the road near the university hospital. Please fix this as soon as possible.",
    "location_address": "Near University Hospital",
    "category": "Infrastructure"
}

print("\n📝 Test Input:")
print(f"   Description: {test_data['text_description'][:60]}...")
print(f"   Location: {test_data['location_address']}")
print(f"   Category: {test_data['category']}")

try:
    print("\n🚀 Sending request to http://localhost:8001/classify-urgency...")
    
    response = requests.post(
        "http://localhost:8001/classify-urgency",
        json=test_data,
        timeout=10
    )
    
    print(f"✅ Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n🎯 Classification Result:")
        print("=" * 50)
        print(f"🚨 Urgency Level: {result['urgency_level']}")
        print(f"📊 Urgency Score: {result['urgency_score']}/10")
        print(f"🎯 Confidence: {result['confidence']:.1%}")
        print(f"🏢 Department: {result['recommended_department']}")
        print(f"⏰ Response Time: {result['estimated_response_time']}")
        
        print(f"\n💭 AI Analysis:")
        print(f"   {result['reasoning']}")
        
        print(f"\n📈 Technical Details:")
        print(f"   📝 Text: {result['text_contribution']:.0%}")
        print(f"   🖼️  Image: {result['image_contribution']:.0%}")
        print(f"   📍 Location: {result['location_context']}")
        print(f"   ⚠️  Safety: {result['safety_context']}")
        
        print("\n" + "=" * 50)
        print("✅ API IS WORKING PERFECTLY!")
        print("🌐 Open http://localhost:8001 in your browser to test the UI")
        print("=" * 50)
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Cannot connect to server")
    print("💡 Make sure the server is running:")
    print("   python start_ui.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"   Type: {type(e).__name__}")
