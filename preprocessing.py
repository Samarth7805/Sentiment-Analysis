
# preprocessing.py
# This file cleans raw text before we use it for training or prediction.
# Steps: lowercase -> fix slang -> remove URLs -> remove punctuation -> remove stopwords

import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# Slang dictionary: maps slang words to normal English words
slang = {
    "lit": "good",
    "fire": "good",
    "sick": "good",
    "dope": "good",
    "goat": "best",
    "mid": "average",
    "trash": "bad",
    "lol": "funny",
    "lmao": "funny",
    "bruh": "",
    "ngl": "honestly",
    "imo": "in my opinion",
    "tbh": "honestly",
    "af": "very",
    "lowkey": "slightly",
    "highkey": "very",
    "slay": "great",
    "bussin": "great",
    "cap": "lie",
    "no cap": "truth",
    "sus": "suspicious",
    "vibe": "feeling",
    "yeet": "throw",
    "periodt": "period",
    "salty": "angry",
    "shade": "insult",
    "ghosted": "ignored",
    "flex": "show off",
    "bet": "okay",
    "fam": "friend",
    "stan": "fan",
    "woke": "aware",
    "clout": "fame",
    "extra": "dramatic",
    "basic": "ordinary",
    "mood": "relatable",
    "tea": "gossip",
    "sis": "sister",
    "bae": "loved one",
    "feels": "emotions",
    "deadass": "seriously",
    "bougie": "fancy",
    "savage": "fierce"
}

negative_emojis = {
    "😞", "😔", "😟", "😕", "🙁", "☹️",
    "😢", "😭", "😩", "😫", "😖", "😣",
    "😠", "😡", "🤬", "😤", "😒", "🙄",
    "👎", "💔", "😓", "🤢", "🤮", "💀",
    "😰", "😨", "😧", "😦", "😥"
}

positive_emojis = {
    "😊", "😄", "😁", "😀", "🙂", "😃",
    "😍", "🥰", "😎", "🤩", "😇", "🥳",
    "❤️", "🧡", "💛", "💚", "💙", "💜",
    "👍", "🔥", "💯", "✨", "🎉", "🎊", "🙌",
    "💪", "👏", "😂", "🤣", "😆"
}


def normalize_slang(text):
    words = text.split()
    result = []
    for word in words:
        if word in slang:
            replacement = slang[word]
            if replacement:
                result.append(replacement)
        else:
            result.append(word)
    return " ".join(result)


def detect_emoji_sentiment(text):
    has_negative = any(e in text for e in negative_emojis)
    has_positive = any(e in text for e in positive_emojis)

    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return None


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = normalize_slang(text)

  
    text = re.sub(r"<.*?>", "", text)

    
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"&lt;|&gt;|&nbsp;|&quot;", " ", text)

    # Remove url starting have http & https, or www
    text = re.sub(r"http\S+|www\S+", "", text)

    # @elonmusk will get  removed
    text = re.sub(r"@\w+", "", text)

    # happy will become happy
    text = re.sub(r"#(\w+)", r"\1", text)


    text = re.sub(r"\s+", " ", text).strip()


    words = text.split()

    clean_words = []

    for w in words:
        if w not in stop_words:
            # Keep word if length > 1 or if it contains any known emoji
            if len(w) > 1 or any(e in w for e in positive_emojis | negative_emojis):
                clean_words.append(w)

    return " ".join(clean_words)


def preprocess_dataframe(df):
    print("Cleaning text data...")
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Remove rows that became empty after cleaning
    df = df[df["cleaned_text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df
