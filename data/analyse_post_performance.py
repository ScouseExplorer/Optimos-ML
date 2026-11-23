
import json
from datetime import datetime, timezone
from collections import Counter

with open("cleaned_linkedin_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

now = datetime.now(timezone.utc)

age_buckets = Counter()
topic_stats = {}
engagement_totals = {
    "likes": 0,
    "comments": 0,
    "shares": 0
}

for post in posts:
    date_string = post.get("published_at")

    if date_string:
        published = datetime.fromisoformat(
            date_string.replace("Z", "+00:00")
        )

        age_hours = (now - published).total_seconds() / 3600

        if age_hours < 1:
            age_buckets["Under 1 hour"] += 1
        elif age_hours < 24:
            age_buckets["1-24 hours"] += 1
        elif age_hours < 168:
            age_buckets["1-7 days"] += 1
        else:
            age_buckets["Over 7 days"] += 1

    topic = post.get("search_topic") or "Unknown"

    if topic not in topic_stats:
        topic_stats[topic] = {
            "posts": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0
        }

    topic_stats[topic]["posts"] += 1

    for metric in engagement_totals:
        value = post["engagement"].get(metric)

        if value is not None:
            engagement_totals[metric] += value
            topic_stats[topic][metric] += value


print("\n--- POST AGE DISTRIBUTION ---")
for bucket, count in age_buckets.items():
    print(f"{bucket}: {count}")

print("\n--- TOTAL ENGAGEMENT ---")
print(engagement_totals)

print("\n--- ENGAGEMENT BY TOPIC ---")
for topic, stats in topic_stats.items():
    print(topic, stats)
