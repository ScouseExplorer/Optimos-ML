
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from statistics import mean

INPUT_FILE = "linkedin_master_dataset_final.json"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    posts = json.load(file)

print("TOTAL POSTS:", len(posts))

# 1. Date coverage
dates = []

for post in posts:
    date = post.get("published_at")

    if date:
        try:
            dates.append(datetime.fromisoformat(
                date.replace("Z", "+00:00")
            ))
        except ValueError:
            pass

print("\n--- DATE COVERAGE ---")

if dates:
    print("Earliest post:", min(dates))
    print("Latest post:", max(dates))

    monthly = Counter(date.strftime("%Y-%m") for date in dates)

    print("\nPosts per month:")
    for month, count in sorted(monthly.items()):
        print(month, count)

# 2. Engagement distribution
likes = []
comments = []
shares = []

for post in posts:
    engagement = post.get("engagement") or {}

    if engagement.get("likes") is not None:
        likes.append(engagement["likes"])

    if engagement.get("comments") is not None:
        comments.append(engagement["comments"])

    if engagement.get("shares") is not None:
        shares.append(engagement["shares"])

print("\n--- ENGAGEMENT ANALYSIS ---")

for name, values in [
    ("Likes", likes),
    ("Comments", comments),
    ("Shares", shares)
]:
    if values:
        print(f"\n{name}")
        print("Average:", round(mean(values), 2))
        print("Median:", sorted(values)[len(values) // 2])
        print("Maximum:", max(values))
        print("Zero values:", sum(value == 0 for value in values))

# 3. Content length
lengths = [
    len(post["text"].strip())
    for post in posts
    if post.get("text") and post["text"].strip()
]

print("\n--- CONTENT QUALITY ---")
print("Posts with text:", len(lengths))
print("Average characters:", round(mean(lengths), 2))
print("Shortest post:", min(lengths))
print("Longest post:", max(lengths))
print("Under 100 characters:", sum(length < 100 for length in lengths))
print("Over 1000 characters:", sum(length > 1000 for length in lengths))

# 4. Topic distribution
print("\n--- TOPIC DISTRIBUTION ---")

topics = Counter(
    post.get("search_topic") or "unknown"
    for post in posts
)

for topic, count in topics.most_common():
    print(f"{topic}: {count}")

print("\n--- ANALYSIS COMPLETE ---")
