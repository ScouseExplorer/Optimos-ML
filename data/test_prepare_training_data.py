import json
import tempfile
import unittest
from pathlib import Path

from data.prepare_training_data import assign_duplicate_groups, prepare_dataset


def make_post(post_id, text, **overrides):
	post = {
		"post_id": post_id,
		"text": text,
		"published_at": "2026-11-19T11:35:13.286Z",
		"engagement": {"likes": 2, "comments": 1, "shares": 0},
		"author": {"id": "author-1", "name": "Author", "type": "company"},
		"search_topic": "Marketing",
	}
	post.update(overrides)
	return post


class PrepareTrainingDataTests(unittest.TestCase):
	def test_prepares_provenance_and_preserves_missing_values(self):
		post = make_post("post-1", "A post with useful evidence.")
		post["engagement"]["likes"] = None
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			source = root / "source.json"
			canonical = root / "canonical.json"
			source.write_text(json.dumps([post]), encoding="utf-8")

			result = prepare_dataset(str(source), str(canonical))
			record = result["records"][0]
			self.assertTrue(canonical.exists())

		self.assertEqual(record["post_id"], "post-1")
		self.assertIsNone(record["engagement"]["likes"])
		self.assertEqual(record["author_type"], "company")
		self.assertEqual(record["search_query"], "Marketing")
		self.assertIsNone(record["metrics_captured_at"])
		self.assertEqual(result["summary"]["missing_metric_values"], 1)

	def test_preserves_optional_forecasting_context(self):
		post = make_post("forecast-context", "A post with complete forecasting context.")
		post["engagement"]["impressions"] = 500
		post["audience_size"] = 10000
		post["prediction_at"] = "2026-11-20T10:00:00Z"
		post["outcome_at"] = "2026-11-27T10:00:00Z"
		post["observation_window_hours"] = 168
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			source = root / "source.json"
			canonical = root / "canonical.json"
			source.write_text(json.dumps([post]), encoding="utf-8")
			result = prepare_dataset(str(source), str(canonical))
			record = result["records"][0]
			self.assertEqual(record["impressions"], 500)
			self.assertEqual(record["audience_size"], 10000)
			self.assertEqual(record["observation_window_hours"], 168)
			self.assertEqual(record["prediction_at"], "2026-11-20T10:00:00.000Z")
			self.assertEqual(record["outcome_at"], "2026-11-27T10:00:00.000Z")

	def test_rejects_bad_records_and_duplicate_ids_with_reasons(self):
		posts = [
			make_post("same-id", "Valid post."),
			make_post("same-id", "Second post with duplicate id."),
			make_post("bad-date", "Invalid date.", published_at="yesterday"),
			make_post("bad-count", "Invalid count.", engagement={"likes": -1}),
		]
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			source = root / "source.json"
			canonical = root / "canonical.json"
			source.write_text(json.dumps(posts), encoding="utf-8")

			result = prepare_dataset(str(source), str(canonical))
			rejected = json.loads((root / "canonical_rejected.json").read_text(encoding="utf-8"))

		self.assertEqual(result["summary"]["canonical_records"], 1)
		self.assertEqual(result["summary"]["rejected_records"], 3)
		self.assertIn("duplicate post_id", rejected[0]["reason"])
		self.assertIn("published_at", rejected[1]["reason"])

	def test_duplicate_groups_join_exact_and_near_matches_only(self):
		records = [
			{"text": "A reliable pipeline validates post metadata and preserves training provenance."},
			{"text": "A reliable pipeline validates post metadata and preserves training provenance."},
			{"text": "A reliable pipeline validates post metadata and preserves training provenance today."},
			{"text": "Unrelated gardening content about soil and flowers."},
		]

		result = assign_duplicate_groups(records, threshold=0.8)

		self.assertEqual(result, {"exact_duplicate_groups": 1, "near_duplicate_groups": 1})
		self.assertEqual(records[0]["duplicate_group_id"], records[1]["duplicate_group_id"])
		self.assertEqual(records[1]["duplicate_group_id"], records[2]["duplicate_group_id"])
		self.assertNotEqual(records[0]["duplicate_group_id"], records[3]["duplicate_group_id"])


if __name__ == "__main__":
	unittest.main()