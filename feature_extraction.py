
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_model, load_model

# Filename for saving the vectorizer
VECTORIZER_FILE = "tfidf_vectorizer.joblib"


def build_tfidf(texts):

    vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"[^\s]+")
    X = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {X.shape}")
    return X, vectorizer


def save_vectorizer(vectorizer):
    save_model(vectorizer, VECTORIZER_FILE)


def load_vectorizer():

    return load_model(VECTORIZER_FILE)


