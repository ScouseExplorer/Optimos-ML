
import json

file_path = "dataset_linkedin-post-search_2026-11-19_11-37-22-869.json"

with open(file_path, "r", encoding="utf-8") as f:
    posts = json.load(f)

zero_posts = []
engaged_posts = []

for post in posts:
    engagement = post.get("engagement") or {}

    likes = engagement.get("likes", 0) or 0
    comments = engagement.get("comments", 0) or 0
    shares = engagement.get("shares", 0) or 0

    if likes == 0 and comments == 0 and shares == 0:
        zero_posts.append(post)
    else:
        engaged_posts.append(post)

print("\n--- FIVE ZERO-ENGAGEMENT POSTS ---")

for post in zero_posts[:5]:
    print(json.dumps({
        "text": post.get("text"),
        "date": post.get("postedAt") or post.get("postedDate"),
        "engagement": post.get("engagement")
    }, indent=2, ensure_ascii=False))

print("\n--- FIVE POSTS WITH ENGAGEMENT ---")

for post in engaged_posts[:5]:
    print(json.dumps({
        "text": post.get("text"),
        "date": post.get("postedAt") or post.get("postedDate"),
        "engagement": post.get("engagement")
    }, indent=2, ensure_ascii=False))
