import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    
    # remove urls
    text = re.sub(r'http\S+', '', text)
    
    # keep numbers
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    words = text.split()
    
    words = [
        lemmatizer.lemmatize(w)
        for w in words if w not in stop_words
    ]
    
    return " ".join(words)


