# src/tests/test_tavily_direct.py

import requests
from src.config import TAVILY_API_KEY


def test_tavily(query: str, n_results: int = 3):
    api_key = TAVILY_API_KEY
    if not api_key:
        print("[ERROR] TAVILY_API_KEY not set in src.config.")
        return

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": n_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    print("\n" + "=" * 80)
    print("QUERY:", query)

    try:
        resp = requests.post(url, json=payload, timeout=15)
        print("HTTP status:", resp.status_code)
        data = resp.json()
    except Exception as exc:
        print("[ERROR] Tavily request failed:", exc)
        return

    if isinstance(data, dict) and "error" in data:
        print("[TAVILY ERROR]:", data["error"])
        return

    results = data.get("results", [])
    print("num results:", len(results))
    if results:
        first = results[0]
        print("top title:", first.get("title"))
        print("top url:", first.get("url"))
        print("content preview:", (first.get("content") or "")[:400].replace("\n", " "))


if __name__ == "__main__":
    test_tavily("Solve d/dx (x^3).", n_results=3)
    test_tavily("linear algebra basics", n_results=3)
