"""Smoke tests for LinkedIn ML intelligence endpoints.

Run this while API server is up:
  python -m uvicorn api.app:app --reload
"""

import json
import requests
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
ML_SERVICE_API_KEY = os.getenv("ML_SERVICE_API_KEY", "")
API_HEADERS = {"X-ML-API-Key": ML_SERVICE_API_KEY} if ML_SERVICE_API_KEY else {}


def pretty(label, payload):
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2)[:2000])


def must_ok(response, name):
    if response.status_code != 200:
        raise RuntimeError(f"{name} failed: {response.status_code} {response.text}")
    return response.json()


def run():
    health = must_ok(requests.get(f"{BASE_URL}/health", timeout=20), "health")
    pretty("health", health)

    post = {
        "content": (
            "I spent 3 months redesigning a payment retry system for failed card transactions. "
            "We reduced failed renewals by 18% and cut support tickets significantly. "
            "Biggest lesson: retries without clear state modeling create hidden failure loops. "
            "What would you optimize first in a billing pipeline? #fintech #backend #payments"
        )
    }

    features = must_ok(requests.post(f"{BASE_URL}/extract-features", headers=API_HEADERS, json=post, timeout=30), "extract-features")
    pretty("extract-features", {"feature_count": features.get("feature_count")})

    classified = must_ok(requests.post(f"{BASE_URL}/classify-content", headers=API_HEADERS, json=post, timeout=30), "classify-content")
    pretty("classify-content", classified)

    analyzed = must_ok(
        requests.post(
            f"{BASE_URL}/analyze-post",
            headers=API_HEADERS,
            json={
                "content": post["content"],
                "historical_posts": [
                    "We shipped a Kubernetes autoscaling improvement last week and learned a lot.",
                    "One API design mistake cost us two sprints. Here is what we changed.",
                ],
            },
            timeout=40,
        ),
        "analyze-post",
    )
    pretty("analyze-post", analyzed)

    user_analysis = must_ok(
        requests.post(
            f"{BASE_URL}/analyze-user",
            headers=API_HEADERS,
            json={
                "user_id": "user-1",
                "historical_posts": [
                    "How we improved API reliability by redesigning retries.",
                    "A painful deployment outage taught me to simplify rollback paths.",
                    "3 lessons from leading a backend migration in fintech.",
                ],
            },
            timeout=40,
        ),
        "analyze-user",
    )
    pretty("analyze-user", user_analysis)

    recommendation = must_ok(
        requests.post(
            f"{BASE_URL}/recommend-content",
            headers=API_HEADERS,
            json={
                "user_id": "user-1",
                "profile": {"industry": "Technology", "role": "Software Engineer", "expertise": ["Cloud", "Fintech", "Backend"]},
                "historical_posts": [
                    "How we improved API reliability by redesigning retries.",
                    "A painful deployment outage taught me to simplify rollback paths.",
                    "3 lessons from leading a backend migration in fintech.",
                ],
                "goal": "increase engagement",
            },
            timeout=40,
        ),
        "recommend-content",
    )
    pretty("recommend-content", recommendation)

    compared = must_ok(
        requests.post(
            f"{BASE_URL}/compare-candidates",
            headers=API_HEADERS,
            json={
                "historical_posts": [
                    "How we improved API reliability by redesigning retries.",
                    "A painful deployment outage taught me to simplify rollback paths.",
                ],
                "candidates": [
                    "5 lessons I learned about AI leadership this week.",
                    "The outage that changed how I design payment retries (and what I fixed).",
                    "A checklist for debugging Kubernetes pod restarts in production.",
                ],
            },
            timeout=60,
        ),
        "compare-candidates",
    )
    pretty("compare-candidates", compared)

    validated = must_ok(
        requests.post(
            f"{BASE_URL}/validate-post",
            headers=API_HEADERS,
            json={
                "content": post["content"],
                "historical_posts": [
                    "How we improved API reliability by redesigning retries.",
                ],
                "recommended_strategy": {
                    "topic": "Payment systems",
                    "format": "personal technical story",
                    "hook_strategy": "question",
                },
            },
            timeout=40,
        ),
        "validate-post",
    )
    pretty("validate-post", validated)

    orchestrated = must_ok(
        requests.post(
            f"{BASE_URL}/intelligence/analyze",
            headers=API_HEADERS,
            json={"content": post["content"], "components": ["content_effectiveness", "recommendation_ranking"]},
            timeout=40,
        ),
        "intelligence/analyze",
    )
    pretty("intelligence/analyze", orchestrated)

    recommendations = must_ok(
        requests.post(
            f"{BASE_URL}/recommendations",
            headers=API_HEADERS,
            json={"profile": {"expertise": ["Cloud"]}, "goal": "help peers", "historical_posts": []},
            timeout=40,
        ),
        "recommendations",
    )
    pretty("recommendations", recommendations)

    evaluation = must_ok(requests.get(f"{BASE_URL}/evaluation/report", headers=API_HEADERS, timeout=20), "evaluation/report")
    pretty("evaluation/report", evaluation)

    internal_health = must_ok(requests.get(f"{BASE_URL}/internal/health", headers=API_HEADERS, timeout=20), "internal/health")
    pretty("internal/health", internal_health)

    print("\nAll intelligence endpoint smoke tests passed.")


if __name__ == "__main__":
    run()
