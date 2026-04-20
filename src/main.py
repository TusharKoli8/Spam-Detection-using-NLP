import pandas as pd

from src.preprocessing import clean_text
from src.vectorization import tfidf_vectorize
from src.analysis import get_top_words
from src.word2vec_demo import train_word2vec, demo_similarity


def run_pipeline():

    print("🔹 Loading dataset...")
    df = pd.read_csv("data/spam.csv", encoding='latin-1')

    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Encode labels
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    print("🔹 Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    # -----------------------------
    # TF-IDF
    # -----------------------------
    print("🔹 Performing TF-IDF...")
    X, vectorizer = tfidf_vectorize(df['clean_text'])

    # -----------------------------
    # Analysis
    # -----------------------------
    print("\n📊 Top Spam Words:")
    spam_words = get_top_words(vectorizer, X, df['label_num'], 1)

    for word, score in spam_words[:15]:
        print(f"{word} → {score:.4f}")

    print("\n📊 Top Ham Words:")
    ham_words = get_top_words(vectorizer, X, df['label_num'], 0)

    for word, score in ham_words[:15]:
        print(f"{word} → {score:.4f}")

    # -----------------------------
    # Word2Vec Demo
    # -----------------------------
    print("\n🧠 Training Word2Vec...")
    w2v_model = train_word2vec()

    print("\n🔍 Word Similarity Demo:")
    demo_similarity(w2v_model)

    print("\n✅ NLP Pipeline Completed!")


if __name__ == "__main__":
    run_pipeline()





