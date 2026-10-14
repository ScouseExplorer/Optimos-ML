# api/app.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import os
import sys
import re
import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.responses import JSONResponse

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
sentence_transformer_model = None

MODEL_VERSIONS = {
    "engagement_predictor": "1.0.0",
    "topic_classifier": "0.1.0",
    "format_classifier": "0.1.0",
    "hook_classifier": "0.1.0",
    "genericity_detector": "0.1.0",
    "recommendation_engine": "0.1.0",
    "semantic_similarity": "0.2.0",
    "feedback_ingestion": "0.1.0",
}

logger = logging.getLogger("linkedin_ml_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ML_SERVICE_API_KEY = os.getenv("ML_SERVICE_API_KEY", "").strip()
AUTH_REQUIRED = bool(ML_SERVICE_API_KEY)
REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-ML-API-Key"
FEEDBACK_PATH = os.path.join(project_root, "data", "feedback_events.jsonl")
os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

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
    
    # Keep the NLP extractor available even when models are not trained yet.
    if nlp_extractor is None:
        try:
            nlp_extractor = NLPFeatureExtractor()
        except Exception as e:
            print(f"⚠️  NLP extractor initialization warning: {e}")

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

class ExtractFeaturesRequest(BaseModel):
    content: str

class ClassifyContentRequest(BaseModel):
    content: str

class AnalyzePostRequest(BaseModel):
    content: str
    user_profile: Optional[Dict[str, Any]] = None
    historical_posts: Optional[List[str]] = None

class ValidatePostRequest(BaseModel):
    content: str
    recommended_strategy: Optional[Dict[str, Any]] = None
    user_profile: Optional[Dict[str, Any]] = None
    historical_posts: Optional[List[str]] = None

class AnalyzeUserRequest(BaseModel):
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    historical_posts: List[str] = Field(default_factory=list)
    performance: Optional[List[Dict[str, Any]]] = None

class RecommendContentRequest(BaseModel):
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    historical_posts: List[str] = Field(default_factory=list)
    historical_performance: Optional[List[Dict[str, Any]]] = None
    global_context: Optional[List[Dict[str, Any]]] = None
    goal: str = "increase engagement"
    optional_topic_request: Optional[str] = None
    candidate_topic: Optional[str] = None

class CompareCandidatesRequest(BaseModel):
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    historical_posts: List[str] = Field(default_factory=list)
    candidates: List[str]


class FeedbackUserEventRequest(BaseModel):
    user_id: str
    recommendation_id: Optional[str] = None
    event_type: str
    metadata: Optional[Dict[str, Any]] = None


class FeedbackPerformanceEventRequest(BaseModel):
    user_id: str
    post_id: str
    content: Optional[str] = None
    metrics: Dict[str, Any]
    predicted: Optional[Dict[str, Any]] = None


def _hash_user_scope(user_id: Optional[str]) -> str:
    if not user_id:
        return "anonymous"
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _get_request_id(request: Request) -> str:
    return request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())


def _is_public_path(path: str) -> bool:
    if path in {"/", "/health", "/docs", "/openapi.json", "/docs/oauth2-redirect", "/redoc"}:
        return True
    return False


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = _get_request_id(request)
    start = perf_counter()

    if AUTH_REQUIRED and not _is_public_path(request.url.path):
        incoming = request.headers.get(API_KEY_HEADER, "")
        if incoming != ML_SERVICE_API_KEY:
            logger.warning(
                "request_denied request_id=%s path=%s method=%s reason=invalid_api_key",
                request_id,
                request.url.path,
                request.method,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized backend request", "request_id": request_id},
                headers={REQUEST_ID_HEADER: request_id},
            )

    response = await call_next(request)
    elapsed_ms = round((perf_counter() - start) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id

    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s latency_ms=%s models_loaded=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        bool(models),
    )
    return response


def _load_sentence_transformer_model():
    global sentence_transformer_model
    if sentence_transformer_model is not None:
        return sentence_transformer_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        return sentence_transformer_model
    except Exception:
        return None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalize_engagement_rate(item: Dict[str, Any]) -> float:
    likes = float(item.get("likes", 0) or 0)
    comments = float(item.get("comments", 0) or 0)
    reposts = float(item.get("reposts", item.get("shares", 0)) or 0)
    followers = float(item.get("followers", item.get("follower_count", 0)) or 0)
    total = likes + comments + reposts

    if followers > 0:
        return _safe_div(total, followers)
    return total


def _classify_topic(features: Dict[str, float], text: str) -> Dict[str, Any]:
    lower = text.lower()
    keyword_sets = {
        "AI": ["ai", "llm", "machine learning", "model", "prompt"],
        "Cloud": ["cloud", "aws", "azure", "kubernetes", "k8s", "docker"],
        "Fintech": ["fintech", "payment", "bank", "trading", "fraud"],
        "Software Engineering": ["backend", "api", "architecture", "system design", "code"],
        "Leadership": ["leadership", "manager", "team", "mentor"],
        "Career": ["career", "interview", "promotion", "job"],
        "Data": ["data", "analytics", "pipeline", "warehouse"],
        "Cybersecurity": ["security", "zero trust", "threat", "vulnerability"],
    }

    scores: Dict[str, float] = {}
    for topic, keywords in keyword_sets.items():
        scores[topic] = float(sum(1 for kw in keywords if kw in lower))

    # Blend with extractor's coarse industry/topic indicators.
    scores["AI"] += float(features.get("trend_ai_ml_score", 0))
    scores["Cloud"] += float(features.get("industry_tech_score", 0))
    scores["Fintech"] += float(features.get("industry_finance_score", 0))

    best_topic = max(scores.items(), key=lambda x: x[1])[0] if scores else "General"
    total_score = sum(scores.values())
    confidence = _clamp(_safe_div(scores.get(best_topic, 0), total_score if total_score > 0 else 1.0))
    if total_score == 0:
        best_topic = "General"
        confidence = 0.25

    return {
        "topic": best_topic,
        "confidence": round(confidence, 2),
        "scores": scores,
        "model": {"name": "topic_classifier", "version": MODEL_VERSIONS["topic_classifier"]},
    }


def _classify_format(features: Dict[str, float], text: str) -> Dict[str, Any]:
    lower = text.lower()
    word_count = features.get("word_count", 0)
    has_question = features.get("has_question", 0) > 0
    bullet_points = features.get("bullet_point_count", 0)
    first_person_ratio = features.get("personal_pronoun_ratio", 0)

    if bullet_points >= 2:
        fmt = "list"
        conf = 0.82
    elif has_question and word_count < 120:
        fmt = "question"
        conf = 0.72
    elif any(k in lower for k in ["i built", "i learned", "my biggest", "we shipped", "i made"]):
        fmt = "personal story"
        conf = 0.79
    elif any(k in lower for k in ["step", "how to", "guide", "tutorial"]):
        fmt = "tutorial"
        conf = 0.75
    elif any(k in lower for k in ["i believe", "opinion", "hot take", "unpopular opinion"]):
        fmt = "opinion"
        conf = 0.71
    elif first_person_ratio > 0.04:
        fmt = "personal technical lesson"
        conf = 0.66
    else:
        fmt = "educational post"
        conf = 0.55

    return {
        "format": fmt,
        "confidence": round(conf, 2),
        "model": {"name": "format_classifier", "version": MODEL_VERSIONS["format_classifier"]},
    }


def _classify_hook(text: str) -> Dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    hook = lines[0] if lines else text.strip()
    lower = hook.lower()

    label = "direct claim"
    confidence = 0.55

    if "?" in hook:
        label = "question"
        confidence = 0.83
    elif any(k in lower for k in ["unpopular opinion", "everyone says", "wrong"]):
        label = "contrarian statement"
        confidence = 0.77
    elif re.search(r"\d+%|\d+x|\d+", hook):
        label = "surprising statistic"
        confidence = 0.74
    elif any(k in lower for k in ["i failed", "mistake", "what i learned", "i learned"]):
        label = "failure"
        confidence = 0.76
    elif any(k in lower for k in ["proud", "excited", "launched", "shipped"]):
        label = "achievement"
        confidence = 0.69
    elif any(k in lower for k in ["i used to", "when i", "last year", "my experience"]):
        label = "personal experience"
        confidence = 0.73

    return {
        "hook_text": hook[:180],
        "hook_type": label,
        "confidence": round(confidence, 2),
        "model": {"name": "hook_classifier", "version": MODEL_VERSIONS["hook_classifier"]},
    }


def _genericity_score(features: Dict[str, float], text: str) -> Dict[str, Any]:
    lower = text.lower()
    words = text.split()
    word_count = len(words)
    unique_ratio = features.get("unique_word_ratio", 0)
    personal_ratio = features.get("personal_pronoun_ratio", 0)
    number_density = _safe_div(len(re.findall(r"\d", text)), max(len(text), 1))

    cliche_terms = [
        "game changer", "unlock your potential", "in today's world", "journey",
        "thought leader", "leverage", "synergy", "best practices"
    ]
    cliche_hits = sum(1 for term in cliche_terms if term in lower)

    score = 0.0
    if word_count < 80:
        score += 0.2
    if unique_ratio < 0.55:
        score += 0.2
    if personal_ratio < 0.01:
        score += 0.2
    if number_density < 0.003:
        score += 0.1
    score += min(0.3, cliche_hits * 0.08)

    genericity = _clamp(score)
    return {
        "genericity": round(genericity, 2),
        "is_generic": genericity >= 0.6,
        "signals": {
            "low_specificity": unique_ratio < 0.55,
            "low_personal_reference_density": personal_ratio < 0.01,
            "cliche_hits": cliche_hits,
            "short_content": word_count < 80,
        },
        "model": {"name": "genericity_detector", "version": MODEL_VERSIONS["genericity_detector"]},
    }


def _jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return _safe_div(len(a_tokens.intersection(b_tokens)), len(a_tokens.union(b_tokens)))


def _tfidf_similarity(a: str, b: str) -> float:
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([a, b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def _semantic_embedding_similarity(a: str, b: str) -> Optional[float]:
    model = _load_sentence_transformer_model()
    if model is None:
        return None
    try:
        emb = model.encode([a, b], normalize_embeddings=True)
        # Cosine similarity for normalized vectors is simple dot product.
        return float(np.dot(emb[0], emb[1]))
    except Exception:
        return None


def _similarity_summary(content: str, historical_posts: Optional[List[str]] = None) -> Dict[str, Any]:
    historical_posts = historical_posts or []
    if not historical_posts:
        return {
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
            "sample_size": 0,
            "most_similar_excerpt": None,
        }

    similarities = []
    method_counts = {"embedding": 0, "tfidf": 0, "jaccard": 0}
    for past in historical_posts:
        if not isinstance(past, str) or not past.strip():
            continue

        emb_sim = _semantic_embedding_similarity(content, past)
        if emb_sim is not None:
            score = emb_sim
            method_counts["embedding"] += 1
        else:
            tfidf_sim = _tfidf_similarity(content, past)
            if tfidf_sim > 0:
                score = tfidf_sim
                method_counts["tfidf"] += 1
            else:
                score = _jaccard_similarity(content, past)
                method_counts["jaccard"] += 1

        similarities.append((score, past))

    if not similarities:
        return {
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
            "sample_size": 0,
            "most_similar_excerpt": None,
            "similarity_method": "none",
        }

    max_sim, top_post = max(similarities, key=lambda x: x[0])
    mean_sim = float(np.mean([s for s, _ in similarities]))
    return {
        "max_similarity": round(max_sim, 3),
        "mean_similarity": round(mean_sim, 3),
        "sample_size": len(similarities),
        "most_similar_excerpt": top_post[:200],
        "similarity_method": max(method_counts.items(), key=lambda x: x[1])[0],
        "method_counts": method_counts,
        "model": {"name": "semantic_similarity", "version": MODEL_VERSIONS["semantic_similarity"]},
    }


def _append_feedback_event(event: Dict[str, Any]) -> None:
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _compute_calibration_delta(predicted: Optional[Dict[str, Any]], metrics: Dict[str, Any]) -> Optional[float]:
    if not predicted:
        return None

    pred = predicted.get("engagement_score")
    if pred is None:
        return None

    likes = float(metrics.get("likes", 0) or 0)
    comments = float(metrics.get("comments", 0) or 0)
    reposts = float(metrics.get("reposts", metrics.get("shares", 0)) or 0)
    followers = float(metrics.get("followers", metrics.get("follower_count", 0)) or 0)
    actual = _safe_div(likes + comments + reposts, followers) * 100 if followers > 0 else likes + comments + reposts
    return round(float(actual - float(pred)), 3)


def _calculate_confidence(sample_size: int, has_model: bool = True) -> float:
    # Confidence rises with sample size and with trained model availability.
    sample_component = min(1.0, _safe_div(sample_size, 50.0))
    model_component = 1.0 if has_model else 0.65
    return round(_clamp(0.25 + 0.55 * sample_component + 0.2 * (model_component - 0.65)), 2)


def _predict_scores_from_features(X_scaled: Optional[np.ndarray]) -> Dict[str, Any]:
    result = {
        "engagement_score": None,
        "viral_probability": None,
        "model": {"name": "engagement_predictor", "version": MODEL_VERSIONS["engagement_predictor"]},
    }
    if X_scaled is None or not models:
        return result

    linear_model = models.get("regression", {}).get("linear")
    viral_model = models.get("classification", {}).get("viral")
    if linear_model is not None:
        result["engagement_score"] = round(float(linear_model.predict(X_scaled)[0]), 2)
    if viral_model is not None:
        result["viral_probability"] = round(float(viral_model.predict_proba(X_scaled)[0][1] * 100), 1)

    return result


def _derive_strengths_weaknesses(features: Dict[str, float]) -> Dict[str, List[str]]:
    strengths: List[str] = []
    weaknesses: List[str] = []

    if features.get("has_question", 0) > 0:
        strengths.append("Includes a question that can stimulate comments")
    else:
        weaknesses.append("No question present for audience interaction")

    if features.get("hashtag_count", 0) >= 3:
        strengths.append("Hashtags are in a healthy range for discoverability")
    else:
        weaknesses.append("Hashtag usage is low; consider 3-5 relevant hashtags")

    if 120 <= features.get("word_count", 0) <= 300:
        strengths.append("Word count is in the common high-engagement range")
    else:
        weaknesses.append("Word count may be suboptimal for LinkedIn engagement")

    if features.get("cta_score", 0) > 0:
        strengths.append("Contains call-to-action language")
    else:
        weaknesses.append("Missing clear call-to-action")

    if features.get("sentiment_polarity", 0) < 0:
        weaknesses.append("Negative tone may reduce engagement for broad audiences")

    return {"strengths": strengths, "weaknesses": weaknesses}


def _analyze_personal_performance(
    historical_posts: List[str],
    performance: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    if not historical_posts:
        return {
            "sample_size": 0,
            "confidence": 0.2,
            "best_topics": [],
            "best_formats": [],
            "weak_topics": [],
            "strong_hooks": [],
        }

    topic_perf: Dict[str, List[float]] = {}
    format_perf: Dict[str, List[float]] = {}
    hook_perf: Dict[str, List[float]] = {}
    fallback_rates: List[float] = []

    for idx, post in enumerate(historical_posts):
        if not post:
            continue
        _, feat = extract_nlp_features(post)
        topic = _classify_topic(feat, post)["topic"]
        fmt = _classify_format(feat, post)["format"]
        hook = _classify_hook(post)["hook_type"]

        if performance and idx < len(performance):
            score = _normalize_engagement_rate(performance[idx])
        else:
            score = (
                feat.get("has_question", 0) * 0.15 +
                min(feat.get("hashtag_count", 0), 5) * 0.05 +
                feat.get("cta_score", 0) * 0.08 +
                0.2
            )
            fallback_rates.append(score)

        topic_perf.setdefault(topic, []).append(score)
        format_perf.setdefault(fmt, []).append(score)
        hook_perf.setdefault(hook, []).append(score)

    def _relative_rank(group: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        if not group:
            return []
        avg = {k: float(np.mean(v)) for k, v in group.items() if v}
        baseline = float(np.mean(list(avg.values()))) if avg else 1.0
        ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "name": k,
                "relative_performance": round(_safe_div(v, baseline if baseline else 1.0), 2),
                "sample_size": len(group[k]),
            }
            for k, v in ranked
        ]

    topic_rank = _relative_rank(topic_perf)
    format_rank = _relative_rank(format_perf)
    hook_rank = _relative_rank(hook_perf)
    sample_size = len([p for p in historical_posts if p])

    return {
        "sample_size": sample_size,
        "confidence": _calculate_confidence(sample_size, has_model=True),
        "best_topics": topic_rank[:3],
        "weak_topics": list(reversed(topic_rank[-2:])) if len(topic_rank) > 2 else [],
        "best_formats": format_rank[:3],
        "strong_hooks": hook_rank[:3],
        "uses_observed_performance": bool(performance),
        "fallback_proxy_used": not bool(performance),
    }


def _build_blueprint(topic: str, fmt: str, hook_type: str) -> Dict[str, Any]:
    base_structure = [
        "hook",
        "specific problem or context",
        "what you did",
        "what changed",
        "lesson",
        "light call-to-action",
    ]

    avoid = [
        "generic motivational language",
        "unsupported claims",
        "buzzword-heavy phrasing without specifics",
    ]

    return {
        "topic": topic,
        "format": fmt,
        "hook_strategy": {
            "type": hook_type,
            "description": f"Lead with a {hook_type} opening tied to real experience",
        },
        "structure": base_structure,
        "recommended_length": "150-250 words",
        "evidence_to_use": [
            "specific project/scenario",
            "measurable result where available",
            "one practical takeaway",
        ],
        "avoid": avoid,
    }


def _strategy_compliance(content: str, strategy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not strategy:
        return {"score": None, "details": "No strategy provided"}

    checks = []
    lower = content.lower()
    topic = str(strategy.get("topic", "")).lower()
    hook_type = str(strategy.get("hook_strategy", strategy.get("hook", ""))).lower()
    fmt = str(strategy.get("format", "")).lower()

    if topic:
        checks.append(1.0 if topic in lower else 0.0)
    if "question" in hook_type:
        checks.append(1.0 if "?" in content else 0.0)
    if "list" in fmt:
        checks.append(1.0 if re.search(r"^\s*(\d+\.|[-*•])", content, flags=re.M) else 0.0)

    if not checks:
        return {"score": None, "details": "Strategy had no machine-checkable fields"}

    return {"score": round(float(np.mean(checks)), 2), "details": "Heuristic strategy compliance"}

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


@app.get("/health")
async def health():
    return {
        "status": "ok" if nlp_extractor is not None else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_loaded": len(models) > 0,
        "model_versions": MODEL_VERSIONS,
        "auth_required": AUTH_REQUIRED,
        "feedback_store": FEEDBACK_PATH,
    }


@app.post("/feedback/user")
async def feedback_user_event(request: FeedbackUserEventRequest):
    request_id = str(uuid.uuid4())
    event = {
        "event_id": request_id,
        "event_family": "user_feedback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_scope": _hash_user_scope(request.user_id),
        "recommendation_id": request.recommendation_id,
        "event_type": request.event_type,
        "metadata": request.metadata or {},
        "model": {"name": "feedback_ingestion", "version": MODEL_VERSIONS["feedback_ingestion"]},
    }
    _append_feedback_event(event)
    return {"status": "recorded", "event_id": request_id}


@app.post("/feedback/performance")
async def feedback_performance_event(request: FeedbackPerformanceEventRequest):
    request_id = str(uuid.uuid4())
    calibration_delta = _compute_calibration_delta(request.predicted, request.metrics)

    event = {
        "event_id": request_id,
        "event_family": "post_performance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_scope": _hash_user_scope(request.user_id),
        "post_id": request.post_id,
        "metrics": request.metrics,
        "predicted": request.predicted or {},
        "calibration_delta": calibration_delta,
        "content_preview": (request.content or "")[:180],
        "model": {"name": "feedback_ingestion", "version": MODEL_VERSIONS["feedback_ingestion"]},
    }
    _append_feedback_event(event)
    return {
        "status": "recorded",
        "event_id": request_id,
        "calibration_delta": calibration_delta,
    }


@app.get("/feedback/stats")
async def feedback_stats():
    if not os.path.exists(FEEDBACK_PATH):
        return {"events": 0, "user_feedback": 0, "performance_feedback": 0}

    total = 0
    user_feedback = 0
    performance_feedback = 0
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            fam = row.get("event_family")
            if fam == "user_feedback":
                user_feedback += 1
            elif fam == "post_performance":
                performance_feedback += 1

    return {
        "events": total,
        "user_feedback": user_feedback,
        "performance_feedback": performance_feedback,
        "storage": FEEDBACK_PATH,
    }


@app.post("/extract-features")
async def extract_features_endpoint(request: ExtractFeaturesRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="content is required")

    _, features = extract_nlp_features(request.content)
    return {
        "feature_count": len(features),
        "features": features,
        "model": {"name": "feature_extractor", "version": "1.0.0"},
    }


@app.post("/classify-content")
async def classify_content(request: ClassifyContentRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="content is required")

    _, features = extract_nlp_features(request.content)
    topic = _classify_topic(features, request.content)
    fmt = _classify_format(features, request.content)
    hook = _classify_hook(request.content)

    return {
        "topic": topic,
        "format": fmt,
        "hook": hook,
    }


@app.post("/analyze-post")
async def analyze_post(request: AnalyzePostRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="content is required")

    X, features = extract_nlp_features(request.content)
    X_scaled = scalers['standard'].transform(X) if models else None

    topic = _classify_topic(features, request.content)
    fmt = _classify_format(features, request.content)
    hook = _classify_hook(request.content)
    genericity = _genericity_score(features, request.content)
    similarity = _similarity_summary(request.content, request.historical_posts)
    prediction = _predict_scores_from_features(X_scaled)
    quality = _derive_strengths_weaknesses(features)

    return {
        "topic": topic,
        "format": fmt,
        "hook": hook,
        "genericity": genericity,
        "similarity": similarity,
        "predicted_engagement": prediction,
        "personal_relevance": round(1.0 - similarity["max_similarity"] * 0.5, 2),
        "confidence": _calculate_confidence(similarity.get("sample_size", 0), has_model=bool(models)),
        "strengths": quality["strengths"],
        "weaknesses": quality["weaknesses"],
        "recommendations": [
            "Add specific evidence from your own experience",
            "Use 3-5 relevant hashtags",
            "Close with a concrete question or CTA",
        ],
    }


@app.post("/validate-post")
async def validate_post(request: ValidatePostRequest):
    analysis = await analyze_post(
        AnalyzePostRequest(
            content=request.content,
            user_profile=request.user_profile,
            historical_posts=request.historical_posts,
        )
    )

    compliance = _strategy_compliance(request.content, request.recommended_strategy)
    unsupported_claim_signal = 0.0
    if request.user_profile:
        expertise = " ".join(request.user_profile.get("expertise", [])) if isinstance(request.user_profile.get("expertise", []), list) else ""
        if expertise and not any(tok in request.content.lower() for tok in expertise.lower().split()[:6]):
            unsupported_claim_signal = 0.4

    return {
        "predicted_engagement": analysis["predicted_engagement"],
        "genericity_score": analysis["genericity"]["genericity"],
        "similarity": analysis["similarity"],
        "personal_relevance": analysis["personal_relevance"],
        "strategy_compliance": compliance,
        "evidence_grounding": {
            "unsupported_generalization_signal": unsupported_claim_signal,
            "note": "Heuristic check; backend should provide evidence map for stronger validation",
        },
        "passes_validation": (
            analysis["genericity"]["genericity"] < 0.65 and
            (compliance["score"] is None or compliance["score"] >= 0.5)
        ),
    }


@app.post("/analyze-user")
async def analyze_user(request: AnalyzeUserRequest):
    analysis = _analyze_personal_performance(request.historical_posts, request.performance)
    return {
        "user_id": request.user_id,
        "personal_performance": analysis,
        "model": {"name": "personal_performance", "version": "0.1.0"},
    }


@app.post("/recommend-content")
async def recommend_content(request: RecommendContentRequest):
    personal = _analyze_personal_performance(request.historical_posts, request.historical_performance)
    sample_size = personal["sample_size"]

    personal_weight = _clamp(_safe_div(sample_size, 40.0), 0.1, 0.8)
    global_weight = round(1.0 - personal_weight, 2)

    best_topic = request.optional_topic_request or request.candidate_topic
    if not best_topic and personal.get("best_topics"):
        best_topic = personal["best_topics"][0]["name"]
    best_topic = best_topic or "Software Engineering"

    best_format = "personal technical story"
    if personal.get("best_formats"):
        best_format = personal["best_formats"][0]["name"]

    hook_type = "personal experience"
    if personal.get("strong_hooks"):
        hook_type = personal["strong_hooks"][0]["name"]

    blueprint = _build_blueprint(best_topic, best_format, hook_type)
    predicted_score = round(0.55 + personal_weight * 0.25, 2)
    confidence = _calculate_confidence(sample_size, has_model=bool(models))

    return {
        "recommendation": {
            "topic": best_topic,
            "angle": f"What I learned while applying {best_topic} in real work",
            "format": best_format,
            "hook_strategy": hook_type,
            "predicted_score": predicted_score,
            "confidence": confidence,
            "sample_size": sample_size,
            "reasons": [
                "Combines personal historical patterns with global priors",
                "Aligned to engagement-oriented structure",
                "Designed for evidence-backed authenticity",
            ],
            "weights": {
                "global_signal": global_weight,
                "personal_signal": round(personal_weight, 2),
            },
        },
        "content_blueprint": blueprint,
        "things_to_avoid": blueprint["avoid"],
        "model": {"name": "recommendation_engine", "version": MODEL_VERSIONS["recommendation_engine"]},
    }


@app.post("/compare-candidates")
async def compare_candidates(request: CompareCandidatesRequest):
    if not request.candidates:
        raise HTTPException(status_code=400, detail="At least one candidate is required")

    scored = []
    for idx, candidate in enumerate(request.candidates):
        analysis = await analyze_post(
            AnalyzePostRequest(
                content=candidate,
                user_profile=request.profile,
                historical_posts=request.historical_posts,
            )
        )
        engagement = analysis["predicted_engagement"].get("engagement_score")
        viral = analysis["predicted_engagement"].get("viral_probability")
        if engagement is None:
            engagement = 50.0
        if viral is None:
            viral = 50.0

        final_score = round(
            0.5 * _clamp(engagement / 100.0) +
            0.3 * _clamp(viral / 100.0) +
            0.2 * (1.0 - analysis["genericity"]["genericity"]),
            3,
        )

        scored.append({
            "candidate_index": idx,
            "content_preview": candidate[:160],
            "score": final_score,
            "analysis": analysis,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "ranked_candidates": scored,
        "best_candidate": scored[0] if scored else None,
        "model": {"name": "candidate_ranker", "version": "0.1.0"},
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
