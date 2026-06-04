
import numpy as np 
from sklearn.linear_model import LogisticRegression
from preprocessing import clean_text, detect_emoji_sentiment
from feature_extraction import load_vectorizer
from utils import load_model

MODEL_FILE = "logistic_regression.joblib"

NEGATION_RULES = {
    "not good": "negative",
    "not great": "negative",
    "not bad": "positive",
    "not happy": "negative",
    "not worth": "negative",
    "not recommend": "negative",
    "no good": "negative",
    "never good": "negative",
    "not terrible": "positive",
    "not awful": "positive",
    "dont like": "negative",
    "don't like": "negative",
    "didnt like": "negative",
    "didn't like": "negative",
    "cant stand": "negative",
    "can't stand": "negative",
    "not satisfied": "negative",
    "not impressed": "negative",
    "not disappointed": "positive",
}

POSITIVE_WORDS = {
    "good", "great", "happy", "awesome", "amazing", "wonderful", "perfect", 
    "love", "excellent", "fantastic", "brilliant", "outstanding", "glad",
    "pleased", "satisfied", "impressed", "worth", "recommend"
}

SARCASM_PATTERNS = [
    "yeah right",
    "oh great",
    "wow thanks",
    "how wonderful",
    "so amazing",
    "just wonderful",
    "oh perfect",
    "totally awesome",
]


def get_model():
    return LogisticRegression(max_iter=1000, random_state=42)


def check_negation(text):

    text_lower = text.lower()
    
    for pattern, sentiment in NEGATION_RULES.items():
        if pattern in text_lower:
            return sentiment
            
    # not + positive word = negative
    words = text_lower.replace(".", " ").replace(",", " ").split()
    negation_words = {"not", "never", "no", "don't", "dont", "didn't", "didnt", "isn't", "isnt", "aren't", "arent", "won't", "wont", "can't", "cant"}
    
    for i in range(len(words) - 1):
        if words[i] in negation_words and words[i+1] in POSITIVE_WORDS:
            return "negative"
            
    return None


def check_sarcasm(text):
   
    text_lower = text.lower()

    for pattern in SARCASM_PATTERNS:
        if pattern in text_lower:
            return True

    if "..." in text:
        positive_words = ["great", "amazing", "wonderful", "love", "perfect", "awesome"]
        for word in positive_words:
            if word in text_lower:
                return True

    return False


def handle_mixed_sentiment(text):
    split_words = ["but ", "however ", "although ", "though "]
    text_lower = text.lower()
    for word in split_words:
        if word in text_lower:
            parts = text_lower.split(word, 1)
            if len(parts) == 2:
                return parts[1].strip()

    return text

def predict_sentiment(text):
    
    original_text = text

    emoji_result = detect_emoji_sentiment(text)

    negation_result = check_negation(text)
    if negation_result:
        return negation_result, 1.0, "Rule: Overridden by Negation matching"

    is_sarcastic = check_sarcasm(text)

    text = handle_mixed_sentiment(text)

    cleaned = clean_text(text)

    vectorizer = load_vectorizer()
    features = vectorizer.transform([cleaned])

    model = load_model(MODEL_FILE)
    prediction = model.predict(features)[0]
    confidence = float(np.max(model.predict_proba(features)))
    reason = "Machine Learning Prediction"

    if is_sarcastic and prediction == "positive":
        prediction = "negative"
        confidence = 1.0
        reason = "Rule: Sarcasm detected, flipped to negative"

    if emoji_result and confidence < 0.6:
        prediction = emoji_result
        confidence = 0.9
        reason = "Rule: High emoji signal shifted sentiment"


    if confidence < 0.45:
        prediction = "neutral"
        reason = "Rule: Low confidence reverted to neutral"

    return prediction, confidence, reason
