
from collections import Counter
import json
from pathlib import Path

INPUT_FILE = "linkedin_posts_third.json"
OUTPUT_FILE = "cleaned_linkedin_posts_third.json"

def process_posts(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    print("\n--- INPUT VERIFICATION ---")
    print("Input file:", path.resolve())
    print("Total raw records:", len(raw_data))

    cleaned_posts = []
    seen_ids = set()
    comments_skipped = 0
    duplicates_skipped = 0

    for post in raw_data:
        if post.get("type") != "post":
            comments_skipped += 1
            continue

        post_id = post.get("id")

        if not post_id or post_id in seen_ids:
            duplicates_skipped += 1
            continue

        seen_ids.add(post_id)

        author = post.get("author") or {}
        engagement = post.get("engagement") or {}
        posted_at = post.get("postedAt") or {}
        article = post.get("article") or {}
        query = post.get("query") or {}

        cleaned_post = {
            "post_id": post_id,
            "text": post.get("content"),
            "author": {
                "name": author.get("name"),
                "type": author.get("type"),
                "id": author.get("id"),
                "position": author.get("position"),
            },
            "published_at": posted_at.get("date"),
            "engagement": {
                "likes": engagement.get("likes"),
                "comments": engagement.get("comments"),
                "shares": engagement.get("shares"),
                "reactions": engagement.get("reactions"),
            },
            "url": post.get("linkedinUrl"),
            "images": post.get("postImages") or [],
            "article": {
                "title": article.get("title"),
                "description": article.get("description"),
                "link": article.get("link"),
            },
            "search_topic": query.get("search"),
            "content_attributes": post.get("contentAttributes") or [],
        }

        cleaned_posts.append(cleaned_post)

    print("Original posts:", len(cleaned_posts))
    print("Comments/non-post records skipped:", comments_skipped)
    print("Duplicates or missing IDs skipped:", duplicates_skipped)

    return cleaned_posts


def analyse_dataset(posts):
    total = len(posts)

    missing_text = sum(not post["text"] for post in posts)
    missing_author = sum(not post["author"]["name"] for post in posts)
    missing_date = sum(not post["published_at"] for post in posts)

    missing_engagement = sum(
        all(post["engagement"][key] is None
            for key in ["likes", "comments", "shares"])
        for post in posts
    )

    all_zero = sum(
        all(post["engagement"][key] == 0
            for key in ["likes", "comments", "shares"])
        for post in posts
        if all(post["engagement"][key] is not None
               for key in ["likes", "comments", "shares"])
    )

    author_types = Counter(
        post["author"]["type"] or "unknown" for post in posts
    )

    topics = Counter(
        post["search_topic"] or "unknown" for post in posts
    )

    print("\n--- DATASET QUALITY REPORT ---")
    print("Total unique posts:", total)
    print("Missing text:", missing_text)
    print("Missing author:", missing_author)
    print("Missing publication date:", missing_date)
    print("Missing engagement:", missing_engagement)
    print("All-zero engagement:", all_zero)

    print("\nAuthor types:")
    print(author_types)

    print("\nSearch topics:")
    print(topics)


if __name__ == "__main__":
    posts = process_posts(INPUT_FILE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=4, ensure_ascii=False)

    analyse_dataset(posts)

    print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")
