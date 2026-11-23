# api/app.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from enum import Enum
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
from intelligence.content.content_effectiveness import ContentEffectivenessEngine
from intelligence.performance.engagement_intelligence import EngagementIntelligenceEngine
from intelligence.performance.performance_prediction import PerformancePredictionEngine
from intelligence.personal.recommendations import recommend_strategy
from intelligence.orchestrator.engine import IntelligenceOrchestrator
from intelligence.orchestrator.registry import build_default_registry
from intelligence.orchestrator.schemas import AnalysisRequest
from intelligence.generative.llm_client import OpenAITextProvider, GenerativeProviderError
from intelligence.generative.refinement import RefinementService
from intelligence.generative.explanation import compare_assessments

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
ML_SERVICE_API_KEY = os.getenv("ML_SERVICE_API_KEY", "").strip()
AUTH_REQUIRED = bool(ML_SERVICE_API_KEY)
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "12000"))
MAX_VARIANTS = int(os.getenv("MAX_VARIANTS", "3"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "10"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

if ENVIRONMENT in {"production", "staging"} and not ML_SERVICE_API_KEY:
    raise RuntimeError("ML_SERVICE_API_KEY is required outside development")

app = FastAPI(
    title="LinkedIn Growth AI",
    description="Evaluate content, summarize observed engagement, and report performance-prediction readiness",
    version="2.1"
)

# Global variables for models
models = {}
scalers = {}
feature_names = []
training_metadata = {}
nlp_extractor = None
sentence_transformer_model = None
content_effectiveness_engine = None
engagement_intelligence_engine = EngagementIntelligenceEngine()
performance_prediction_engine = PerformancePredictionEngine(training_metadata)

MODEL_VERSIONS = {
    "content_effectiveness": "1.0.0",
    "engagement_intelligence": "1.0.0",
    "performance_prediction": "0.1.0",
    "topic_classifier": "0.1.0",
    "format_classifier": "0.1.0",
    "hook_classifier": "0.1.0",
    "genericity_detector": "0.1.0",
    "semantic_similarity": "0.2.0",
    "feedback_ingestion": "0.1.0",
}

logger = logging.getLogger("linkedin_ml_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-ML-API-Key"
FEEDBACK_PATH = os.path.join(project_root, "data", "feedback_events.jsonl")
os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

# OpenAI client
openai_client = None
refinement_service = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        openai_client = OpenAI(api_key=api_key)
        refinement_service = RefinementService(OpenAITextProvider(openai_client))
        print("OpenAI API initialized")
    else:
        print("OpenAI API key not found. AI refinement will be disabled.")
        print("   Add OPENAI_API_KEY to .env file to enable.")
except Exception as e:
    print(f"OpenAI initialization warning: {e}")

def load_trained_models():
    """Load pre-trained models and their training metadata from disk."""
    global models, scalers, feature_names, training_metadata, nlp_extractor
    global content_effectiveness_engine, performance_prediction_engine
    
    model_path = os.path.join(project_root, 'models', 'linkedin_models.pkl')
    
    # Keep the NLP extractor available even when models are not trained yet.
    if nlp_extractor is None:
        try:
            nlp_extractor = NLPFeatureExtractor()
        except Exception as e:
            print(f"NLP extractor initialization warning: {e}")

    if not os.path.exists(model_path):
        print(f"Trained models not found at {model_path}")
        print("Run 'python train_models.py' first to train models on your data")
        return False
    
    try:
        print("Loading trained models...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        models['regression'] = model_data['regression_models']
        models['classification'] = model_data['classification_models']
        scalers['standard'] = model_data['scaler']
        feature_names = model_data['feature_names']
        training_metadata = model_data.get('training_metadata', {})
        performance_prediction_engine = PerformancePredictionEngine(training_metadata)
        
        # Initialize NLP feature extractor
        nlp_extractor = NLPFeatureExtractor()
        content_effectiveness_engine = ContentEffectivenessEngine(nlp_extractor)
        
        print("Models loaded successfully!")
        print(f"  - Regression models: {list(models['regression'].keys())}")
        print(f"  - Classification models: {list(models['classification'].keys())}")
        print(f"  - Features: {len(feature_names)} NLP features")
        print(f"  - Training samples: {training_metadata.get('training_samples', 'unknown')}")
        print(f"  - Engagement target: {training_metadata.get('engagement_target', 'unknown')}")
        
        return True
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Load models on startup
if not load_trained_models():
    print("\n" + "="*60)
    print("MODELS NOT LOADED - API will not work properly")
    print("="*60)
    print("\nTo fix this:")
    print("1. Run: python train_models.py")
    print("2. Wait for training to complete")
    print("3. Restart this API server")
    print("="*60 + "\n")

# Pydantic models
class PostRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)

class PredictionResponse(BaseModel):
    content_preview: str
    predictions: Dict
    top_features: List[Dict]
    optimization_suggestions: List[str]

class ExtractFeaturesRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)

class ClassifyContentRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)

class AnalyzePostRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)
    user_profile: Optional[Dict[str, Any]] = None
    historical_posts: Optional[List[Any]] = None
    intended_topic: Optional[str] = None

class ValidatePostRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)
    recommended_strategy: Optional[Dict[str, Any]] = None
    user_profile: Optional[Dict[str, Any]] = None
    historical_posts: Optional[List[str]] = None

class AnalyzeUserRequest(BaseModel):
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    historical_posts: List[Any] = Field(default_factory=list)
    performance: Optional[List[Dict[str, Any]]] = None

class RecommendContentRequest(BaseModel):
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    historical_posts: List[Any] = Field(default_factory=list)
    historical_performance: Optional[List[Dict[str, Any]]] = None
    global_context: Optional[List[Dict[str, Any]]] = None
    goal: str = "increase engagement"
    optional_topic_request: Optional[str] = None
    candidate_topic: Optional[str] = None

class CompareCandidatesRequest(BaseModel):
    user_id: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    historical_posts: List[Any] = Field(default_factory=list)
    candidates: List[str] = Field(..., max_length=MAX_CANDIDATES)
    intended_topic: Optional[str] = None

class IntelligenceAnalyzeRequest(BaseModel):
    account_id: Optional[str] = Field(default=None, description="Opaque application account identifier; never infer this from public author names")
    content: Optional[str] = None
    intended_topic: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    goal: Optional[str] = None
    historical_posts: List[Any] = Field(default_factory=list)
    historical_performance: List[Dict[str, Any]] = Field(default_factory=list)
    global_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    components: Optional[List[str]] = None


class RecommendationFeedbackType(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    used = "used"
    helpful = "helpful"

class FeedbackUserEventRequest(BaseModel):
    user_id: str
    recommendation_id: Optional[str] = None
    event_type: RecommendationFeedbackType
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
    """Calibration is disabled until a validated numeric performance target exists."""
    return None


def _analyze_personal_performance(
    historical_posts: List[Any],
    performance: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    identified_posts = [post for post in historical_posts if isinstance(post, dict)]
    return engagement_intelligence_engine.analyze(identified_posts, performance)


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
        "message": "LinkedIn Growth AI",
        "version": "2.0",
        "engines": {
            "content_effectiveness": "ready" if content_effectiveness_engine else "unavailable",
            "engagement_intelligence": "ready",
            "performance_prediction": performance_prediction_engine.readiness()["status"],
        },
        "training_artifacts_available": len(models) > 0,
        "features": len(feature_names),
        "status": "ready" if content_effectiveness_engine else "degraded"
    }


@app.get("/internal/health")
async def internal_health():
	return {
		"liveness": "ok",
		"readiness": "ready" if content_effectiveness_engine is not None and nlp_extractor is not None else "not_ready",
		"authentication_required": AUTH_REQUIRED,
		"performance_prediction": performance_prediction_engine.readiness()["status"],
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

@app.get("/evaluation/report")
async def evaluation_report():
	from intelligence.evaluation.monitoring import build_evaluation_report
	return build_evaluation_report(
		dataset_version="linkedin-master-2949-v1",
		component_versions={
			"content_effectiveness": "1.0.0",
			"engagement_intelligence": "1.0.0",
			"personal_intelligence": "1.0.0",
			"recommendations": "1.0.0",
			"forecasting": "0.1.0",
		},
		content={"status": "heuristic", "rubric_version": "content-rubric-1.0", "human_assessment_available": False, "notes": ["Human agreement evaluation is pending."]},
		performance={"status": "descriptive", "missing_value_proportion": None, "observation_consistency": "captured-at unavailable in legacy dataset", "notes": ["The legacy model is not a forecasting result."]},
		personal={"status": "account_baselines_available", "minimum_observation_count": 5, "baseline_stability": None, "notes": ["Public observations are excluded."]},
		recommendations={"status": "deterministic", "acceptance_count": None, "helpful_count": None, "relevance_sample_count": None, "notes": ["Feedback collection is not yet a durable production store."]},
		forecasting={"status": "not_ready", "baseline_comparison": None, "prediction_error": None, "calibration": None, "drift": None, "model_version": None, "notes": ["No validated forecasting model is enabled."]},
	)

@app.get("/health")
async def health():
    return {
        "status": "ok" if nlp_extractor is not None else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engines": {
            "content_effectiveness": "ready" if content_effectiveness_engine else "unavailable",
            "engagement_intelligence": "ready",
            "performance_prediction": performance_prediction_engine.readiness()["status"],
        },
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
    if content_effectiveness_engine is None:
        raise HTTPException(status_code=503, detail="Content effectiveness engine is unavailable")

    assessment = content_effectiveness_engine.evaluate(request.content)
    features = assessment['features']
    topic = _classify_topic(features, request.content)
    fmt = _classify_format(features, request.content)
    hook = _classify_hook(request.content)

    return {
        "topic": topic,
        "format": fmt,
        "hook": hook,
        "content_effectiveness": assessment,
    }


@app.post("/analyze-post")
async def analyze_post(request: AnalyzePostRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    if content_effectiveness_engine is None:
        raise HTTPException(status_code=503, detail="Content effectiveness engine is unavailable")

    assessment = content_effectiveness_engine.evaluate(request.content, request.intended_topic)
    features = assessment['features']

    topic = _classify_topic(features, request.content)
    fmt = _classify_format(features, request.content)
    hook = _classify_hook(request.content)
    genericity = _genericity_score(features, request.content)
    similarity = _similarity_summary(request.content, request.historical_posts)
    prediction = performance_prediction_engine.readiness()

    return {
        "topic": topic,
        "format": fmt,
        "hook": hook,
        "genericity": genericity,
        "similarity": similarity,
        "content_effectiveness": assessment,
        "performance_prediction": prediction,
        "personal_relevance": round(1.0 - similarity["max_similarity"] * 0.5, 2),
        "strengths": assessment["strengths"],
        "weaknesses": assessment["weaknesses"],
        "recommendations": assessment["weaknesses"],
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
        "performance_prediction": analysis["performance_prediction"],
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
        "unidentified_posts_ignored": sum(not isinstance(post, dict) for post in request.historical_posts),
        "model": {"name": "engagement_intelligence_engine", "version": "1.0.0"},
    }


@app.post("/recommendations")
@app.post("/recommend-content")
async def recommend_content(request: RecommendContentRequest):
    profile = dict(request.profile or {})
    if request.optional_topic_request:
        profile.setdefault("expertise", [request.optional_topic_request])
    elif request.candidate_topic:
        profile.setdefault("expertise", [request.candidate_topic])
    recommendation = recommend_strategy(
        profile=profile,
        goal=request.goal,
        historical_posts=request.historical_posts,
        historical_performance=request.historical_performance,
        global_evidence=request.global_context,
    )
    topic = recommendation.get("topic")
    blueprint = _build_blueprint(
        topic,
        recommendation["format"],
        recommendation["hook_strategy"],
    ) if topic else None

    return {
        "recommendation": recommendation,
        "content_blueprint": blueprint,
        "things_to_avoid": recommendation["things_to_avoid"],
        "performance_prediction": performance_prediction_engine.readiness(),
        "model": {"name": "evidence_aware_recommendation_engine", "version": "1.0.0"},
    }


@app.post("/compare-candidates")
async def compare_candidates(request: CompareCandidatesRequest):
    if not request.candidates:
        raise HTTPException(status_code=400, detail="At least one candidate is required")
    if content_effectiveness_engine is None:
        raise HTTPException(status_code=503, detail="Content effectiveness engine is unavailable")

    scored = []
    for idx, candidate in enumerate(request.candidates):
        assessment = content_effectiveness_engine.evaluate(candidate, request.intended_topic)

        scored.append({
            "candidate_index": idx,
            "content_preview": candidate[:160],
            "score": assessment["content_effectiveness_score"],
            "score_type": assessment["score_type"],
            "analysis": assessment,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "ranked_candidates": scored,
        "best_candidate": scored[0] if scored else None,
        "ranking_basis": "content_characteristics_only",
        "model": {"name": "content_effectiveness_engine", "version": "1.0.0"},
    }

@app.post("/intelligence/analyze")
async def intelligence_analyze(request: IntelligenceAnalyzeRequest):
    """Run only the intelligence components relevant to the supplied inputs."""
    orchestrator = IntelligenceOrchestrator(build_default_registry(
        content_engine=content_effectiveness_engine,
        engagement_engine=engagement_intelligence_engine,
        prediction_engine=performance_prediction_engine,
    ))
    try:
        return orchestrator.analyze(AnalysisRequest(**request.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/predict/engagement", response_model=PredictionResponse)
async def predict_engagement(request: PostRequest):
    """Report prediction readiness separately from content effectiveness."""
    if content_effectiveness_engine is None:
        return {
            "content_preview": request.content[:50] + "...",
            "predictions": performance_prediction_engine.readiness(),
            "top_features": [],
            "optimization_suggestions": []
        }
    assessment = content_effectiveness_engine.evaluate(request.content)
    readiness = performance_prediction_engine.readiness()
    return {
        "content_preview": request.content[:100] + "...",
        "predictions": readiness,
        "content_effectiveness": assessment,
        "top_features": [
            {"dimension": name, "score": score}
            for name, score in assessment["dimension_scores"].items()
        ],
        "optimization_suggestions": assessment["weaknesses"][:5]
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
    info['training_metadata'] = training_metadata
    
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
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)
    style: Optional[str] = "professional"  # professional, casual, inspirational
    variants: Optional[int] = Field(default=1, ge=1, le=MAX_VARIANTS)

class RefineResponse(BaseModel):
    original: str
    refined: List[str]
    analysis: Dict
    improvements: List[str]
    rubric_comparison: Dict

@app.post("/refine/post", response_model=RefineResponse)
async def refine_post(request: RefineRequest):
    """
    Refine a LinkedIn post using AI based on content-rubric evidence
    
    This endpoint:
    1. Analyzes content characteristics
    2. Identifies what's missing (questions, hashtags, etc.)
    3. Uses GPT to rewrite it with improvements
    4. Returns both original and refined versions
    """
    
    if refinement_service is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured. Add OPENAI_API_KEY to .env file."
        )
    
    if content_effectiveness_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Content effectiveness engine is unavailable."
        )
    
    # Step 1: Analyze text characteristics; performance forecasting is separate.
    original_assessment = content_effectiveness_engine.evaluate(request.content)
    features_dict = original_assessment['features']
    predictions = {
        "content_effectiveness_score": original_assessment['content_effectiveness_score'],
        "score_type": original_assessment['score_type'],
        "performance_prediction": performance_prediction_engine.readiness(),
    }
    
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

CURRENT CONTENT ANALYSIS:
- Content rubric score: {predictions['content_effectiveness_score']} (not a performance forecast)
- Performance prediction: unavailable until comparable impression- and audience-labelled outcomes are collected
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
4. Improves clarity and structure without predicting or claiming engagement outcomes
5. Uses line breaks for readability
6. Is between 150-250 words
7. Includes 3-5 relevant hashtags at the end
8. Includes a question or call-to-action

IMPORTANT: Return ONLY the improved post text. No explanations, no metadata, just the post content."""

    # Step 4: Generate bounded refined variants through the provider abstraction.
    try:
        refined_versions = refinement_service.refine_variants(
            original=request.content,
            style=style_tone,
            assessment=original_assessment,
            improvements=improvements_needed,
            variant_count=request.variants,
        )
        
        # Step 5: Compare content characteristics without forecasting outcomes.
        rubric_comparison = {}
        for idx, refined in enumerate(refined_versions):
            refined_assessment = content_effectiveness_engine.evaluate(refined)
            rubric_comparison[f'version_{idx+1}'] = compare_assessments(original_assessment, refined_assessment)
        
        return {
            "original": request.content,
            "refined": refined_versions,
            "analysis": predictions,
            "improvements": improvements_needed,
            "rubric_comparison": rubric_comparison
        }
    
    except GenerativeProviderError:
        raise HTTPException(
            status_code=502,
            detail="The content generation provider did not return a usable completion.",
        )
    except Exception:
        logger.exception("unexpected_refinement_error")
        raise HTTPException(
            status_code=500,
            detail="Content refinement could not be completed.",
        )

@app.post("/refine/compare")
async def compare_versions(request: PostRequest):
    """
    Compare original vs AI-refined version side-by-side
    Useful for A/B testing before posting
    """
    
    if refinement_service is None:
        raise HTTPException(status_code=503, detail="OpenAI API not configured")
    
    if content_effectiveness_engine is None:
        raise HTTPException(status_code=503, detail="Content effectiveness engine is unavailable")
    original_assessment = content_effectiveness_engine.evaluate(request.content)
    
    # Generate refined version (single, best version)
    refine_response = await refine_post(RefineRequest(content=request.content, variants=1))
    
    refined_text = refine_response["refined"][0]
    refined_assessment = content_effectiveness_engine.evaluate(refined_text)
    rubric_comparison = refine_response["rubric_comparison"]["version_1"]
    
    return {
        "comparison": {
            "original": {
                "text": request.content,
                "content_effectiveness_score": original_assessment['content_effectiveness_score'],
                "score_type": original_assessment['score_type'],
                "word_count": original_assessment['structure']['word_count'],
                "hashtags": int(original_assessment['features'].get('hashtag_count', 0))
            },
            "refined": {
                "text": refined_text,
                "content_effectiveness_score": refined_assessment['content_effectiveness_score'],
                "score_type": refined_assessment['score_type'],
                "rubric_comparison": rubric_comparison
            }
        },
        "recommendation": "Review the refined text against your goals; rubric scores are not performance forecasts.",
        "performance_prediction": performance_prediction_engine.readiness(),
        "improvements_made": refine_response["improvements"]
    }
    

if __name__ == "__main__":
    import uvicorn
    print("\nStarting LinkedIn Growth AI API...")
    print("Performance prediction: not ready until validated labels and a model exist")
    print("OpenAI refinement: " + ("Enabled" if openai_client else "Disabled (add API key)"))
    print("📍 API will be available at: http://localhost:8000")
    print("📖 Interactive docs at: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
