import os
import unittest

from fastapi.testclient import TestClient


class Phase11ApiContractTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		os.environ["ENVIRONMENT"] = "development"
		os.environ["ML_SERVICE_API_KEY"] = "phase11-test-key"
		from api.app import app, ML_SERVICE_API_KEY
		cls.client = TestClient(app)
		cls.headers = {"X-ML-API-Key": ML_SERVICE_API_KEY}

	def test_expected_phase11_routes_exist(self):
		from api.app import app
		paths = {route.path for route in app.routes}
		for path in (
			"/classify-content",
			"/analyze-post",
			"/compare-candidates",
			"/analyze-user",
			"/recommendations",
			"/recommend-content",
			"/predict/engagement",
			"/internal/health",
			"/health",
		):
			self.assertIn(path, paths)

	def test_protected_route_rejects_missing_key(self):
		response = self.client.post("/classify-content", json={"content": "A post"})
		self.assertEqual(response.status_code, 401)

	def test_recommendations_alias_returns_structured_contract(self):
		response = self.client.post(
			"/recommendations",
			headers=self.headers,
			json={
				"profile": {"expertise": ["Fintech"]},
				"goal": "help peers",
				"historical_posts": [],
			},
		)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertIn("recommendation", payload)
		self.assertIn("performance_prediction", payload)
		self.assertIn("model", payload)

	def test_orchestrator_response_contract_is_structured(self):
		response = self.client.post(
			"/intelligence/analyze",
			headers=self.headers,
			json={"content": "What I learned about APIs?\n\nWe changed our retry policy.", "components": ["content_effectiveness"]},
		)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertIn("status", payload)
		self.assertIn("components", payload)
		self.assertIn("evidence", payload)
		self.assertIn("limitations", payload)
		component = payload["components"]["content_effectiveness"]
		for key in ("component", "version", "status", "result", "evidence", "limitations", "execution_ms", "error"):
			self.assertIn(key, component)

	def test_internal_readiness_separates_liveness(self):
		response = self.client.get("/internal/health", headers=self.headers)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["liveness"], "ok")
		self.assertIn(payload["readiness"], ("ready", "not_ready"))
		self.assertIn("performance_prediction", payload)

	def test_empty_candidates_are_rejected(self):
		response = self.client.post(
			"/compare-candidates",
			headers=self.headers,
			json={"candidates": []},
		)
		self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
	unittest.main()
