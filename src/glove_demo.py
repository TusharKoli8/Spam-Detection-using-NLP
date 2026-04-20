import numpy as np

# small demo embeddings (simulated)
fake_glove = {
    "win": np.array([1.0, 0.9]),
    "won": np.array([0.95, 0.88]),
    "prize": np.array([0.9, 0.85]),
    "free": np.array([0.92, 0.89]),
    "gift": np.array([0.88, 0.84]),
    "hello": np.array([0.1, 0.2])
}

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def demo_glove():
    pairs = [
        ("win", "won"),
        ("win", "prize"),
        ("free", "gift"),
        ("win", "hello")
    ]

    for w1, w2 in pairs:
        if w1 in fake_glove and w2 in fake_glove:
            sim = cosine_similarity(fake_glove[w1], fake_glove[w2])
            print(f"Similarity({w1}, {w2}) = {sim:.2f}")