# inspect_qdrant_collection.py (at repo root or under src/tests/)

from src.tools.qdrant_tool import get_qdrant


def main():
    q = get_qdrant()
    info = q.client.get_collection("hendrycks_math")
    print("status:", info.status)

    # Newer Qdrant clients expose points_count
    print("points_count:", getattr(info, "points_count", "N/A"))

    # Optionally inspect config to see vector params
    print("config:", info.config)


if __name__ == "__main__":
    main()
