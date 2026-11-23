import json


def extract_linkedin_posts(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            posts = json.load(f)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return []

    if isinstance(posts, dict):
        for key in ["data", "results", "posts"]:
            if isinstance(posts.get(key), list):
                posts = posts[key]
                break
        else:
            posts = [posts]

    extracted_data = []

    for post in posts:
        if not isinstance(post, dict):
            continue

        engagement = post.get("engagement") or {}

        cleaned_post = {
            "text": post.get("text") or post.get("comment"),
            "author": (
                post.get("author", {}).get("name")
                if isinstance(post.get("author"), dict)
                else post.get("author")
            ),
            "likes": engagement.get("likes", post.get("likesCount")),
            "comments": engagement.get("comments", post.get("commentsCount")),
            "shares": engagement.get("shares"),
            "reactions": engagement.get("reactions", []),
            "url": post.get("postUrl") or post.get("url"),
            "date": post.get("postedAt") or post.get("postedDate")
        }

        extracted_data.append(cleaned_post)

    print(f"Successfully extracted {len(extracted_data)} posts.")
    return extracted_data


def analyse_engagement(posts):
    total = len(posts)
    all_zero = 0
    has_engagement = 0
    missing = 0

    for post in posts:
        likes = post.get("likes")
        comments = post.get("comments")
        shares = post.get("shares")

        if likes is None and comments is None and shares is None:
            missing += 1
        elif (likes or 0) == 0 and (comments or 0) == 0 and (shares or 0) == 0:
            all_zero += 1
        else:
            has_engagement += 1

    print("\n--- ENGAGEMENT ANALYSIS ---")
    print("Total posts:", total)
    print("All-zero engagement:", all_zero)
    print("Posts with engagement:", has_engagement)
    print("Missing engagement:", missing)


if __name__ == "__main__":
    file_path = "dataset_linkedin-post-search_2026-11-19_11-37-22-869.json"
    posts_data = extract_linkedin_posts(file_path)

    if posts_data:
        analyse_engagement(posts_data)

        with open("cleaned_linkedin_posts.json", "w", encoding="utf-8") as f:
            json.dump(posts_data, f, indent=4, ensure_ascii=False)

        print("Saved cleaned dataset.")