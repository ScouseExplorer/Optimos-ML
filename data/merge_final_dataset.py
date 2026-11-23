
import json
from pathlib import Path
from collections import Counter

MASTER_FILE = "linkedin_master_dataset.json"
THIRD_FILE = "cleaned_linkedin_posts_third.json"
OUTPUT_FILE = "linkedin_master_dataset_final.json"


def load_dataset(filename):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Load both datasets
master_posts = load_dataset(MASTER_FILE)
third_posts = load_dataset(THIRD_FILE)

print("Existing master dataset:", len(master_posts))
print("Third dataset:", len(third_posts))

# Merge using post IDs
merged_posts = {}
duplicates = 0
missing_text = 0

for post in master_posts + third_posts:

    post_id = post.get("post_id")
    text = post.get("text")

    if not post_id:
        continue

    if not text or not text.strip():
        missing_text += 1
        continue

    if post_id in merged_posts:
        duplicates += 1

        # Prefer the newer third-dataset record
        if post in third_posts:
            merged_posts[post_id] = post
    else:
        merged_posts[post_id] = post


final_posts = list(merged_posts.values())

# Save final dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(final_posts, file, indent=4, ensure_ascii=False)

# Report
print("\n--- FINAL MASTER DATASET REPORT ---")
print("Previous master posts:", len(master_posts))
print("Third dataset posts:", len(third_posts))
print("Duplicate IDs found:", duplicates)
print("Records without text skipped:", missing_text)
print("Final unique posts:", len(final_posts))
print("Remaining to reach 2,000:", max(0, 2000 - len(final_posts)))

print("\nAuthor distribution:")
print(Counter(
    post["author"].get("type", "unknown")
    for post in final_posts
))

print("\nTopic distribution:")
print(Counter(
    post.get("search_topic", "unknown")
    for post in final_posts
))

print(f"\nSaved final dataset to: {OUTPUT_FILE}")
