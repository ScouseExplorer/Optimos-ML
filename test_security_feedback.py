"""In-process checks for auth middleware, feedback ingestion, and similarity metadata."""

import os
from fastapi.testclient import TestClient


# Enable auth before importing app.
os.environ["ML_SERVICE_API_KEY"] = "local-test-key"

from api.app import app  # noqa: E402


client = TestClient(app)


def test_public_health_is_accessible():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "auth_required" in data


def test_protected_endpoint_requires_key():
    response = client.post("/extract-features", json={"content": "hello"})
    assert response.status_code == 401


def test_protected_endpoint_with_key_works():
    response = client.post(
        "/extract-features",
        headers={"X-ML-API-Key": "local-test-key"},
        json={"content": "I built a payment API in fintech. What would you improve?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] > 0


def test_feedback_endpoints():
    headers = {"X-ML-API-Key": "local-test-key"}

    user_feedback = client.post(
        "/feedback/user",
        headers=headers,
        json={
            "user_id": "user-123",
            "recommendation_id": "rec-1",
            "event_type": "helpful",
            "metadata": {"note": "felt authentic"},
        },
    )
    assert user_feedback.status_code == 200

    perf_feedback = client.post(
        "/feedback/performance",
        headers=headers,
        json={
            "user_id": "user-123",
            "post_id": "post-99",
            "content": "I shared a failure in our migration and what we changed.",
            "predicted": None,
            "metrics": {
                "likes": 120,
                "comments": 20,
                "reposts": 10,
                "followers": 10000,
            },
        },
    )
    assert perf_feedback.status_code == 200

    stats = client.get("/feedback/stats", headers=headers)
    assert stats.status_code == 200
    data = stats.json()
    assert data["events"] >= 2


def test_similarity_metadata_present():
    headers = {"X-ML-API-Key": "local-test-key"}
    response = client.post(
        "/analyze-post",
        headers=headers,
        json={
            "content": "Building resilient billing retries changed our churn profile.",
            "historical_posts": [
                "How we designed resilient billing retries for subscription recovery.",
                "Kubernetes debugging lessons from a production incident.",
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "similarity" in data
    assert "similarity_method" in data["similarity"]


if __name__ == "__main__":
    test_public_health_is_accessible()
    test_protected_endpoint_requires_key()
    test_protected_endpoint_with_key_works()
    test_feedback_endpoints()
    test_similarity_metadata_present()
    print("security/feedback tests passed")
