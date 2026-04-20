import numpy as np

def get_top_words(vectorizer, X, label_series, label_value, top_n=20):

    # FIX: convert to numpy boolean array
    mask = (label_series == label_value).values

    # Apply mask
    X_filtered = X[mask]

    # average tfidf score
    avg_tfidf = np.mean(X_filtered, axis=0).A1

    words = vectorizer.get_feature_names_out()

    word_scores = list(zip(words, avg_tfidf))

    # sort descending
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

    return sorted_words[:top_n]