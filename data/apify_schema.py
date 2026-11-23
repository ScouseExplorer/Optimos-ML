
import json

file_path = "dataset_linkedin-post-search_2026-11-19_11-37-22-869.json"

with open(file_path, "r", encoding="utf-8") as f:
    posts = json.load(f)

print("TOTAL POSTS:", len(posts))

print("\n--- FIRST POST: COMPLETE STRUCTURE ---")

print(json.dumps(
    posts[0],
    indent=4,
    ensure_ascii=False
))
