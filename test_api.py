# test_api.py
"""Test the LinkedIn Growth AI API"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("1. Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction():
    """Test the engagement prediction endpoint"""
    print("\n" + "="*60)
    print("2. Testing Engagement Prediction")
    print("="*60)
    
    test_post = {
        "content": """Excited to share our latest AI breakthrough! 🚀
        
After months of research, our team has developed a new approach to machine learning 
that reduces training time by 40%. This could revolutionize how we build AI systems.

What are your thoughts on the future of AI? #AI #MachineLearning #Innovation #Tech"""
    }
    
    print(f"\nTest Post:\n{test_post['content'][:100]}...\n")
    
    response = requests.post(f"{BASE_URL}/predict/engagement", json=test_post)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Predictions:")
        for key, value in result['predictions'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n🔥 Top 5 Impactful Features:")
        for i, feat in enumerate(result['top_features'][:5], 1):
            print(f"  {i}. {feat['feature']}: {feat['value']} (impact: {feat['impact']})")
        
        print(f"\n💡 Optimization Suggestions:")
        for i, suggestion in enumerate(result['optimization_suggestions'], 1):
            print(f"  {i}. {suggestion}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*60)
    print("3. Testing Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if 'top_10_features' in result:
            print(f"\n🏆 Top 10 Most Important Features:")
            for i, feat in enumerate(result['top_10_features'], 1):
                effect = "↑" if feat['coefficient'] > 0 else "↓"
                print(f"  {i}. {effect} {feat['feature']}: {feat['coefficient']:.3f}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    print("\n" + "="*60)
    print("🚀 LinkedIn Growth AI - API Testing")
    print("="*60)
    print("Testing API at:", BASE_URL)
    
    try:
        # Test 1: Health check
        if not test_health_check():
            print("\n❌ Health check failed!")
            return
        
        # Test 2: Prediction
        if not test_prediction():
            print("\n❌ Prediction test failed!")
            return
        
        # Test 3: Model info
        test_model_info()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\n📖 Visit http://localhost:8000/docs for interactive API documentation")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API!")
        print("Make sure the server is running:")
        print("  python -m uvicorn api.app:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
