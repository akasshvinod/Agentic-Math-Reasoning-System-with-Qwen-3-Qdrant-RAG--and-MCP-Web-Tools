# src/tests/test_embedder.py

from src.embedder.embedder import get_embedder


def test_embedder():
    embedder = get_embedder()
    text = "What is the derivative of sin(x)?"
    vec = embedder.embed(text)

    print("OK: got embedding")
    print("Model:", embedder.model_name)
    print("Dim from object:", embedder.dim)
    print("Vector length:", len(vec))
    print("First 8 dims:", vec[:8])


if __name__ == "__main__":
    test_embedder()
