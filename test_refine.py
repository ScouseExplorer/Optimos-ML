# test_refine.py
"""Test the OpenAI refinement endpoint"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_refine_post():
    """Test the AI refinement feature"""
    
    print("\n" + "="*80)
    print("🤖 Testing OpenAI Post Refinement")
    print("="*80)
    
    # Test post with issues
    test_post = {
        "content": """Building AI and training AI are two very different battles
AI app developers focus on usability, performance, and user experience.AI model trainers focus on data quality, bias control, and model behavior.Developers optimize interfaces and features for real-world use.Trainers optimize learning processes inside the model.Great AI happens only when both worlds work together.But only one needs therapy after training runs.""",
        "style": "professional",
        "variants": 2
    }
    
    print(f"\n📝 Original Post ({len(test_post['content'])} chars):")
    print("-" * 80)
    print(test_post['content'])
    print("-" * 80)
    
    try:
        print("\n🔄 Sending to API for refinement...")
        response = requests.post(f"{BASE_URL}/refine/post", json=test_post, timeout=30)
        
        if response.status_code == 503:
            print("\n❌ OpenAI API not configured!")
            print("\nSetup steps:")
            print("1. Get API key from: https://platform.openai.com/api-keys")
            print("2. Copy .env.example to .env")
            print("3. Add your API key to .env file")
            print("4. Restart the API server")
            return False
        
        if response.status_code != 200:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        # Show analysis
        print("\n📊 Original Post Analysis:")
        print(f"  • Engagement Score: {result['analysis'].get('linear_score', 0):.1f}/100")
        print(f"  • Viral Probability: {result['analysis'].get('viral_probability', 0):.1f}%")
        
        print(f"\n🔍 Issues Found ({len(result['improvements'])} problems):")
        for i, issue in enumerate(result['improvements'], 1):
            print(f"  {i}. {issue}")
        
        # Show refined versions
        for idx, refined in enumerate(result['refined'], 1):
            improvement = result['estimated_improvement'][f'version_{idx}']
            
            print(f"\n{'='*80}")
            print(f"✨ Refined Version {idx}")
            print(f"{'='*80}")
            print(refined)
            print("-" * 80)
            print(f"📈 Predicted Performance:")
            print(f"  • Engagement Score: {improvement['engagement_score']:.1f}/100 ({improvement['improvement']:+.1f})")
            print(f"  • Viral Probability: {improvement['viral_probability']:.1f}%")
            print(f"  • Improvement: {improvement['improvement']:.1f} points")
        
        print("\n" + "="*80)
        print("✅ Refinement test completed successfully!")
        print("="*80)
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API!")
        print("Make sure the server is running:")
        print("  C:/Users/akinl/LinkedIn-AI/.venv/Scripts/python.exe -m uvicorn api.app:app --reload")
        return False
    except requests.exceptions.Timeout:
        print("\n⏱️  Request timed out. OpenAI might be slow. Try again.")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_compare():
    """Test the comparison endpoint"""
    
    print("\n" + "="*80)
    print("⚖️  Testing Side-by-Side Comparison")
    print("="*80)
    
    test_post = {
        "content": "Just launched our new product. Check it out!"
    }
    
    print(f"\n📝 Original: {test_post['content']}")
    
    try:
        response = requests.post(f"{BASE_URL}/refine/compare", json=test_post, timeout=30)
        
        if response.status_code == 503:
            print("\n⚠️  OpenAI API not configured (skipping this test)")
            return False
        
        if response.status_code != 200:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        comp = result['comparison']
        
        print("\n" + "="*80)
        print("ORIGINAL")
        print("="*80)
        print(f"Text: {comp['original']['text']}")
        print(f"Engagement: {comp['original']['engagement_score']:.1f}")
        print(f"Viral Prob: {comp['original']['viral_probability']:.1f}%")
        
        print("\n" + "="*80)
        print("REFINED")
        print("="*80)
        print(f"Text: {comp['refined']['text']}")
        print(f"Engagement: {comp['refined']['engagement_score']:.1f} ({comp['refined']['improvement']:+.1f})")
        print(f"Viral Prob: {comp['refined']['viral_probability']:.1f}%")
        
        print(f"\n🎯 Recommendation: {result['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("🧪 LinkedIn Growth AI - OpenAI Integration Test Suite")
    print("="*80)
    print("\nTesting API at:", BASE_URL)
    print("Make sure the server is running first!\n")
    
    # Test refinement
    test_refine_post()
    
    # Uncomment to test comparison
    # test_compare()
    
    print("\n💡 Tips:")
    print("  • Try different styles: 'professional', 'casual', 'inspirational'")
    print("  • Generate multiple variants to choose the best one")
    print("  • Use /refine/compare for quick before/after view")
    print("\n📖 Full docs at: http://localhost:8000/docs\n")

if __name__ == "__main__":
    main()
