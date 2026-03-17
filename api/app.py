# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()  # This will load from .env in the current working directory

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nlp.feature_extractor import NLPFeatureExtractor

app = FastAPI(
    title="LinkedIn Growth AI - Real Data Edition",
    description="Predict LinkedIn post engagement using ML models trained on 1,608 real posts + AI refinement",
    version="2.1"
)

# Global variables for models
models = {}
scalers = {}
feature_names = []
nlp_extractor = None

# OpenAI client
openai_client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        openai_client = OpenAI(api_key=api_key)
        print("✓ OpenAI API initialized")
    else:
        print("⚠️  OpenAI API key not found. AI refinement will be disabled.")
        print("   Add OPENAI_API_KEY to .env file to enable.")
except Exception as e:
    print(f"⚠️  OpenAI initialization warning: {e}")

def load_trained_models():
    """Load pre-trained models from disk (trained on 1,608 LinkedIn posts)"""
    global models, scalers, feature_names, nlp_extractor
    
    model_path = os.path.join(project_root, 'models', 'linkedin_models.pkl')
    
    if not os.path.exists(model_path):
        print(f"⚠️  Trained models not found at {model_path}")
        print("⚠️  Run 'python train_models.py' first to train models on your data")
        return False
    
    try:
        print("Loading trained models...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        models['regression'] = model_data['regression_models']
        models['classification'] = model_data['classification_models']
        scalers['standard'] = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Initialize NLP feature extractor
        nlp_extractor = NLPFeatureExtractor()
        
        print("✓ Models loaded successfully!")
        print(f"  - Regression models: {list(models['regression'].keys())}")
        print(f"  - Classification models: {list(models['classification'].keys())}")
        print(f"  - Features: {len(feature_names)} NLP features")
        print(f"  - Trained on: 1,608 real LinkedIn posts")
        
        return True
    
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

# Load models on startup
if not load_trained_models():
    print("\n" + "="*60)
    print("⚠️  MODELS NOT LOADED - API will not work properly")
    print("="*60)
    print("\nTo fix this:")
    print("1. Run: python train_models.py")
    print("2. Wait for training to complete")
    print("3. Restart this API server")
    print("="*60 + "\n")

# Pydantic models
class PostRequest(BaseModel):
    content: str

class PredictionResponse(BaseModel):
    content_preview: str
    predictions: Dict
    top_features: List[Dict]
    optimization_suggestions: List[str]

# Helper functions
def extract_nlp_features(text: str) -> np.ndarray:
    """Extract all NLP features from text"""
    if nlp_extractor is None:
        raise ValueError("NLP extractor not initialized")
    
    # Extract features using the same extractor used in training
    features_dict = nlp_extractor.extract_all_features(text)
    
    # Convert to array in the correct feature order
    X = np.array([[features_dict.get(f, 0) for f in feature_names]])
    
    return X, features_dict

# API Endpoints
@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "message": "LinkedIn Growth AI - Trained on 1,608 Real Posts",
        "version": "2.0",
        "models_loaded": len(models) > 0,
        "regression_models": list(models.get('regression', {}).keys()) if models else [],
        "classification_models": list(models.get('classification', {}).keys()) if models else [],
        "features": len(feature_names),
        "status": "ready" if models else "models not loaded"
    }

@app.post("/predict/engagement", response_model=PredictionResponse)
async def predict_engagement(request: PostRequest):
    """Predict engagement for a LinkedIn post using NLP features"""
    
    if not models or nlp_extractor is None:
        return {
            "content_preview": request.content[:50] + "...",
            "predictions": {"error": "Models not loaded. Run train_models.py first."},
            "top_features": [],
            "optimization_suggestions": []
        }
    
    # Extract NLP features from text
    X, features_dict = extract_nlp_features(request.content)
    
    # Scale features
    X_scaled = scalers['standard'].transform(X)
    
    # Make predictions
    predictions = {}
    
    # Regression predictions (engagement scores)
    for name, model in models['regression'].items():
        score = float(model.predict(X_scaled)[0])
        predictions[f'{name}_engagement_score'] = round(score, 2)
    
    # Classification predictions (probabilities)
    for name, model in models['classification'].items():
        proba = model.predict_proba(X_scaled)[0][1]
        predictions[f'{name}_probability'] = round(proba * 100, 1)  # Convert to percentage
        predictions[f'is_{name}'] = bool(model.predict(X_scaled)[0])
    
    # Get top features affecting this prediction
    linear_model = models['regression']['linear']
    coef = linear_model.coef_
    
    # Calculate contribution of each feature
    feature_contributions = []
    for i, fname in enumerate(feature_names):
        contribution = float(coef[i] * X_scaled[0][i])
        feature_contributions.append({
            "feature": fname,
            "value": round(float(features_dict.get(fname, 0)), 2),
            "impact": round(contribution, 2)
        })
    
    # Sort by absolute impact
    feature_contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
    top_features = feature_contributions[:10]
    
    # Generate optimization suggestions
    suggestions = []
    
    if features_dict.get('has_question', 0) == 0:
        suggestions.append("💡 Add a question to boost engagement (e.g., 'What's your experience with this?')")
    
    if features_dict.get('hashtag_count', 0) < 3:
        suggestions.append("💡 Add 2-3 relevant hashtags to increase discoverability")
    
    if features_dict.get('word_count', 0) < 100:
        suggestions.append("💡 Expand your content to 150-250 words for better engagement")
    
    if features_dict.get('sentiment_polarity', 0) < 0:
        suggestions.append("💡 Use more positive language to increase engagement")
    
    if features_dict.get('cta_score', 0) == 0:
        suggestions.append("💡 Add a call-to-action: 'Share', 'Comment', or 'Tag someone'")
    
    return {
        "content_preview": request.content[:100] + "...",
        "predictions": predictions,
        "top_features": top_features,
        "optimization_suggestions": suggestions[:5]
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the trained models"""
    if not models:
        return {"error": "Models not loaded"}
    
    # Read model info from file
    info_path = os.path.join(project_root, 'models', 'model_info.txt')
    feature_importance_path = os.path.join(project_root, 'ml', 'models', 'feature_importance.csv')
    
    info = {}
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info['training_info'] = f.read()
    
    # Get top features
    if os.path.exists(feature_importance_path):
        import pandas as pd
        df = pd.read_csv(feature_importance_path)
        top_10 = df.head(10).to_dict('records')
        info['top_10_features'] = top_10
    
    return info

# ============================================================================
# AI REFINEMENT ENDPOINTS (OpenAI Integration)
# ============================================================================

class RefineRequest(BaseModel):
    content: str
    style: Optional[str] = "professional"  # professional, casual, inspirational
    variants: Optional[int] = 1  # Number of versions to generate (1-3)

class RefineResponse(BaseModel):
    original: str
    refined: List[str]
    analysis: Dict
    improvements: List[str]
    estimated_improvement: Dict

@app.post("/refine/post", response_model=RefineResponse)
async def refine_post(request: RefineRequest):
    """
    Refine a LinkedIn post using AI based on ML model predictions
    
    This endpoint:
    1. Analyzes your post with ML models
    2. Identifies what's missing (questions, hashtags, etc.)
    3. Uses GPT to rewrite it with improvements
    4. Returns both original and refined versions
    """
    
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured. Add OPENAI_API_KEY to .env file."
        )
    
    if not models or nlp_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Run train_models.py first."
        )
    
    # Step 1: Analyze the original post with ML model
    X, features_dict = extract_nlp_features(request.content)
    X_scaled = scalers['standard'].transform(X)
    
    # Get predictions
    predictions = {}
    for name, model in models['regression'].items():
        score = float(model.predict(X_scaled)[0])
        predictions[f'{name}_score'] = round(score, 2)
    
    for name, model in models['classification'].items():
        proba = model.predict_proba(X_scaled)[0][1]
        predictions[f'{name}_probability'] = round(proba * 100, 1)
    
    # Step 2: Build improvement context for AI
    improvements_needed = []
    
    if features_dict.get('has_question', 0) == 0:
        improvements_needed.append("Add a compelling question to drive engagement")
    
    if features_dict.get('hashtag_count', 0) < 3:
        improvements_needed.append(f"Add {3 - int(features_dict.get('hashtag_count', 0))} more relevant hashtags")
    
    if features_dict.get('word_count', 0) < 100:
        improvements_needed.append(f"Expand content from {int(features_dict.get('word_count', 0))} to 150-250 words")
    elif features_dict.get('word_count', 0) > 300:
        improvements_needed.append("Condense content to 200-300 words for better engagement")
    
    if features_dict.get('sentiment_polarity', 0) < 0:
        improvements_needed.append("Use more positive, uplifting language")
    
    if features_dict.get('cta_score', 0) == 0:
        improvements_needed.append("Add a call-to-action (e.g., 'Share your thoughts', 'Tag someone')")
    
    if features_dict.get('engagement_keyword_score', 0) < 2:
        improvements_needed.append("Include engagement keywords like 'insight', 'excited', 'proud', 'share'")
    
    # Step 3: Create AI prompt with data-driven insights
    style_instructions = {
        "professional": "professional and authoritative tone, suitable for executives and thought leaders",
        "casual": "conversational and friendly tone, relatable and approachable",
        "inspirational": "motivational and uplifting tone, inspiring action and growth"
    }
    
    style_tone = style_instructions.get(request.style, style_instructions["professional"])
    
    prompt = f"""You are a LinkedIn content optimization expert. Improve this LinkedIn post based on data-driven insights.

ORIGINAL POST:
{request.content}

CURRENT PERFORMANCE ANALYSIS:
- Engagement Score: {predictions.get('linear_score', 0)}/100
- Viral Probability: {predictions.get('viral_probability', 0)}%
- Word Count: {int(features_dict.get('word_count', 0))}
- Hashtags: {int(features_dict.get('hashtag_count', 0))}
- Has Question: {'Yes' if features_dict.get('has_question', 0) else 'No'}
- Sentiment: {'Positive' if features_dict.get('sentiment_polarity', 0) > 0 else 'Negative'}

KEY IMPROVEMENTS NEEDED:
{chr(10).join(f'• {imp}' for imp in improvements_needed)}

YOUR TASK:
Create an improved version of this LinkedIn post that:
1. Keeps the original message and authenticity
2. Applies the improvements listed above
3. Uses a {style_tone}
4. Optimizes for LinkedIn engagement (saves, shares, comments)
5. Uses line breaks for readability
6. Is between 150-250 words
7. Includes 3-5 relevant hashtags at the end
8. Includes a question or call-to-action

IMPORTANT: Return ONLY the improved post text. No explanations, no metadata, just the post content."""

    # Step 4: Generate refined versions with OpenAI
    try:
        refined_versions = []
        for i in range(min(request.variants, 3)):
            model_to_use = os.getenv("OPENAI_MODEL", "gpt-5")
            
            print(f"\n[DEBUG] Calling OpenAI with:")
            print(f"  Model: {model_to_use}")
            print(f"  Max tokens: {os.getenv('OPENAI_MAX_TOKENS', 500)}")
            print(f"  Prompt length: {len(prompt)} characters")
            
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert LinkedIn content writer who creates engaging, high-performing posts."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 500))
            )
            
            print(f"\n[DEBUG] Response received:")
            print(f"  Response object: {response}")
            print(f"  Choices: {response.choices}")
            print(f"  Message content: '{response.choices[0].message.content}'")
            print(f"  Content length: {len(response.choices[0].message.content)} characters")
            
            refined_post = response.choices[0].message.content.strip()
            
            # Check if response is empty
            if not refined_post:
                print(f"\n[WARNING] Model returned empty response for variant {i+1}")
                print(f"Full response: {response}")
                # Fallback: return the original content if empty
                refined_post = request.content
            else:
                print(f"[SUCCESS] Refined post generated ({len(refined_post)} chars)")
            
            refined_versions.append(refined_post)
        
        # Step 5: Predict engagement for refined version(s)
        estimated_improvements = {}
        for idx, refined in enumerate(refined_versions):
            X_refined, features_refined = extract_nlp_features(refined)
            X_refined_scaled = scalers['standard'].transform(X_refined)
            
            new_score = float(models['regression']['linear'].predict(X_refined_scaled)[0])
            new_viral_prob = float(models['classification']['viral'].predict_proba(X_refined_scaled)[0][1] * 100)
            
            estimated_improvements[f'version_{idx+1}'] = {
                'engagement_score': round(new_score, 2),
                'viral_probability': round(new_viral_prob, 1),
                'improvement': round(new_score - predictions.get('linear_score', 0), 2)
            }
        
        return {
            "original": request.content,
            "refined": refined_versions,
            "analysis": predictions,
            "improvements": improvements_needed,
            "estimated_improvement": estimated_improvements
        }
    
    except Exception as e:
        print(f"Error during refinement: {str(e)}")
        print(f"Error type: {type(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: Error code: {str(e)}"
        )

@app.post("/refine/compare")
async def compare_versions(request: PostRequest):
    """
    Compare original vs AI-refined version side-by-side
    Useful for A/B testing before posting
    """
    
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI API not configured")
    
    # Get analysis for original
    X_orig, features_orig = extract_nlp_features(request.content)
    X_orig_scaled = scalers['standard'].transform(X_orig)
    
    orig_score = float(models['regression']['linear'].predict(X_orig_scaled)[0])
    orig_viral = float(models['classification']['viral'].predict_proba(X_orig_scaled)[0][1] * 100)
    
    # Generate refined version (single, best version)
    refine_response = await refine_post(RefineRequest(content=request.content, variants=1))
    
    refined_text = refine_response["refined"][0]
    refined_est = refine_response["estimated_improvement"]["version_1"]
    
    return {
        "comparison": {
            "original": {
                "text": request.content,
                "engagement_score": round(orig_score, 2),
                "viral_probability": round(orig_viral, 1),
                "word_count": int(features_orig.get('word_count', 0)),
                "hashtags": int(features_orig.get('hashtag_count', 0))
            },
            "refined": {
                "text": refined_text,
                "engagement_score": refined_est['engagement_score'],
                "viral_probability": refined_est['viral_probability'],
                "improvement": refined_est['improvement']
            }
        },
        "recommendation": "Use refined version" if refined_est['improvement'] > 5 else "Both versions are similar",
        "improvements_made": refine_response["improvements"]
    }
    

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting LinkedIn Growth AI API...")
    print("📊 Models trained on 1,608 real LinkedIn posts")
    print("🤖 OpenAI refinement: " + ("Enabled" if openai_client else "Disabled (add API key)"))
    print("📍 API will be available at: http://localhost:8000")
    print("📖 Interactive docs at: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
