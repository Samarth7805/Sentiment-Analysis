"""
utils.py
This file helps the whole project by doing 3 things:
1. Load the dataset from train.csv
2. Save a trained model to a file
3. Load a saved model from a file
"""

import os
import joblib
import pandas as pd

MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def load_dataset():
    s140_cols = ['polarity', 'id', 'date', 'query', 'user', 'text']
    s140_map = {0: 'negative', 2: 'neutral', 4: 'positive'}

    # 1. Standard files
    df1 = pd.read_csv("train.csv", encoding="latin-1")[["text", "sentiment"]]
    df2 = pd.read_csv("test.csv", encoding="latin-1")[["text", "sentiment"]]
    df3 = pd.read_csv("emoji_dataset.csv")[["text", "sentiment"]]

    # 2. Sentiment140 files
    df4 = pd.read_csv("testdata.manual.2009.06.14.csv", encoding="latin-1", names=s140_cols)
    df4['sentiment'] = df4['polarity'].map(s140_map)

    df5 = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", names=s140_cols)
    df5['sentiment'] = df5['polarity'].map(s140_map)
    df5 = df5.dropna(subset=['text']).groupby('sentiment').sample(n=25000, random_state=42)

    # 3. Combine & Clean
    df = pd.concat([df1, df2, df3, df4[["text", "sentiment"]], df5[["text", "sentiment"]]], ignore_index=True)
    
    df = df.dropna(subset=["text"])
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)

    print("Total rows loaded:", len(df))
    return df


def save_model(model, filename):

    path = MODEL_DIR + "/" + filename
    joblib.dump(model, path)



_model_cache = {}

def load_model(filename):
    global _model_cache
    if filename not in _model_cache:
        path = MODEL_DIR + "/" + filename
        _model_cache[filename] = joblib.load(path)
    return _model_cache[filename]
