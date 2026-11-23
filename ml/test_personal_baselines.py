import unittest

from intelligence.personal.personal_intelligence import analyze_personal_baselines


def classify(content):
	return {"topic": "AI" if "AI" in content else "Cloud", "format": "list", "hook": "question"}


class PersonalBaselineTests(unittest.TestCase):
	def test_account_id_and_post_ids_are_required(self):
		with self.assertRaises(ValueError):
			analyze_personal_baselines("", [], [])
		with self.assertRaises(ValueError):
			analyze_personal_baselines("account-1", [{"content": "Post"}], [])
		with self.assertRaises(ValueError):
			analyze_personal_baselines("account-1", [], [{"likes": 1}])

	def test_sparse_personal_history_is_not_ranked(self):
		posts = [{"post_id": str(index), "content": "AI post", "published_at": "2026-11-21T10:00:00Z"} for index in range(4)]
		performance = [{"post_id": str(index), "likes": index + 1, "comments": 0, "shares": 0, "impressions": 100} for index in range(4)]
		result = analyze_personal_baselines("account-1", posts, performance, classify=classify)
		self.assertEqual(result["evidence_status"], "insufficient_personal_history")
		self.assertEqual(result["observed_post_count"], 4)
		self.assertEqual(result["baselines"], {})

	def test_five_observations_create_account_bound_distributions(self):
		posts = [{"post_id": str(index), "content": "AI post", "published_at": f"2026-11-{index + 1:02d}T10:00:00Z"} for index in range(5)]
		performance = [{"post_id": str(index), "likes": index + 1, "comments": index, "shares": 0, "impressions": 100} for index in range(5)]
		result = analyze_personal_baselines("account-1", posts, performance, classify=classify)
		self.assertEqual(result["evidence_status"], "sufficient_personal_history")
		self.assertEqual(result["observed_post_count"], 5)
		self.assertEqual(result["baselines"]["likes"]["median"], 3)
		self.assertEqual(result["baselines"]["comments"]["count"], 5)
		self.assertEqual(result["account_id"], "account-1")
		self.assertTrue(result["group_comparisons"]["topic"]["AI"]["eligible_for_comparison"])
		self.assertTrue(any("Public observations are not included" in limitation for limitation in result["limitations"]))

	def test_missing_counts_preserve_missing_values(self):
		posts = [{"post_id": str(index), "content": "AI post"} for index in range(5)]
		performance = [{"post_id": str(index), "likes": index + 1, "comments": None, "shares": 0, "impressions": 100} for index in range(5)]
		result = analyze_personal_baselines("account-1", posts, performance, classify=classify)
		self.assertEqual(result["evidence_status"], "sufficient_personal_history")
		self.assertEqual(result["baselines"]["comments"]["count"], 0)
		self.assertIsNone(result["baselines"]["engagement_total"]["mean"])

	def test_records_from_a_different_account_are_rejected(self):
		posts = [{"post_id": "a", "account_id": "account-2", "content": "AI"}]
		performance = [{"post_id": "a", "account_id": "account-1", "likes": 1, "comments": 1, "shares": 0}]
		with self.assertRaises(ValueError):
			analyze_personal_baselines("account-1", posts, performance, classify=classify)

	def test_duplicates_are_rejected(self):
		posts = [{"post_id": "a", "content": "AI"}, {"post_id": "a", "content": "AI"}]
		with self.assertRaises(ValueError):
			analyze_personal_baselines("account-1", posts, [])
		with self.assertRaises(ValueError):
			analyze_personal_baselines("account-1", [{"post_id": "a", "content": "AI"}], [{"post_id": "a"}, {"post_id": "a"}])


if __name__ == "__main__":
	unittest.main()
