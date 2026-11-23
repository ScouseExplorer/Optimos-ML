
import json
from pathlib import Path
from collections import Counter

OLD_FILE = "cleaned_linkedin_posts.json"
NEW_FILE = "cleaned_linkedin_posts_new.json"
OUTPUT_FILE = "linkedin_master_dataset.json"


def load_dataset(filename):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find {filename}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Load both datasets
old_posts = load_dataset(OLD_FILE)
new_posts = load_dataset(NEW_FILE)

print("OLD DATASET:", len(old_posts))
print("NEW DATASET:", len(new_posts))

# Merge and deduplicate using LinkedIn post IDs
master_posts = {}

for post in old_posts:
    post_id = post.get("post_id")
    if post_id:
        master_posts[post_id] = post

old_unique_count = len(master_posts)

for post in new_posts:
    post_id = post.get("post_id")
    if post_id:
        # Prefer the newer scrape if the same post appears twice
        master_posts[post_id] = post

combined_posts = list(master_posts.values())

cross_duplicates = (
    old_unique_count + len(new_posts) - len(combined_posts)
)

# Save master dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(combined_posts, file, indent=4, ensure_ascii=False)

# Summary
print("\n--- MASTER DATASET REPORT ---")
print("Previous dataset:", len(old_posts))
print("New dataset:", len(new_posts))
print("Cross-dataset duplicates:", cross_duplicates)
print("Final unique posts:", len(combined_posts))
print("Remaining to reach 2,000:", max(0, 2000 - len(combined_posts)))

print("\nAuthor distribution:")
print(Counter(
    post["author"]["type"] or "unknown"
    for post in combined_posts
))

print("\nTopic distribution:")
print(Counter(
    post["search_topic"] or "unknown"
    for post in combined_posts
))

print("\nSaved:", OUTPUT_FILE)
