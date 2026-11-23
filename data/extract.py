
import json
from collections import Counter

file_path = "dataset_linkedin-post-search_2026-11-19_11-37-22-869.json"

with open(file_path, "r", encoding="utf-8") as f:
    records = json.load(f)

print("TOTAL RECORDS:", len(records))

print("\nRECORD TYPES:")
print(Counter(record.get("type") for record in records))

comments = []
posts = []

for record in records:
    record_type = record.get("type")

    if record_type == "comment":
        comments.append({
            "record_type": "comment",
            "text": record.get("commentary"),
            "author": (record.get("actor") or {}).get("name"),
            "date": record.get("createdAt"),
            "engagement": record.get("engagement"),
            "original_post_id": record.get("postId"),
            "original_post_url": (record.get("query") or {}).get("post")
        })

    elif record_type == "post":
        posts.append(record)

print("\nCOMMENTS:", len(comments))
print("POSTS:", len(posts))

with open("linkedin_comments.json", "w", encoding="utf-8") as f:
    json.dump(comments, f, indent=4, ensure_ascii=False)

with open("linkedin_posts.json", "w", encoding="utf-8") as f:
    json.dump(posts, f, indent=4, ensure_ascii=False)

print("\nExport completed.")
