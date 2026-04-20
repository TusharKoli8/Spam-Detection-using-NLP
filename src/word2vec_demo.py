from gensim.models import Word2Vec
import pandas as pd
from src.preprocessing import clean_text

def train_word2vec():

    df = pd.read_csv("data/spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # preprocess
    df['clean_text'] = df['text'].apply(clean_text)

    sentences = [text.split() for text in df['clean_text']]

    # train small model
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    return model


def demo_similarity(model):

    words = ["free", "win", "call", "claim"]

    for word in words:
        if word in model.wv:
            print(f"\nSimilar to '{word}':")
            print(model.wv.most_similar(word, topn=5))