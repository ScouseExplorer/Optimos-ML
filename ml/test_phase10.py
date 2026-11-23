import os
import unittest

from fastapi.testclient import TestClient


class Phase10Tests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		os.environ["ENVIRONMENT"] = "development"
		os.environ["ML_SERVICE_API_KEY"] = "phase10-test-key"

	def test_health_is_public_and_reports_evaluation_engine(self):
		from api.app import app
		client = TestClient(app)
		response = client.get("/health")
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertIn("engines", payload)
		self.assertIn("performance_prediction", payload["engines"])

	def test_evaluation_report_is_versioned(self):
		from api.app import app
		client = TestClient(app)
		from api.app import ML_SERVICE_API_KEY
		response = client.get("/evaluation/report", headers={"X-ML-API-Key": ML_SERVICE_API_KEY})
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["evaluation_version"], "1.0")
		self.assertEqual(payload["dataset_version"], "linkedin-master-2949-v1")
		self.assertEqual(payload["forecasting"]["status"], "not_ready")

	def test_refine_variant_count_is_bounded_by_request_schema(self):
		from api.app import RefineRequest
		with self.assertRaises(ValueError):
			RefineRequest(content="A post", variants=99)
		request = RefineRequest(content="A post", variants=2)
		self.assertEqual(request.variants, 2)


if __name__ == "__main__":
	unittest.main()
