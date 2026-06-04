# Sentiment Analyzer

A text classification project built with Logistic Regression, TF-IDF, and Streamlit. It predicts whether a piece of text is **Positive**, **Negative**, or **Neutral** — and explains *why*.

---

## What it does

You paste in a tweet, a review, or any text. The app runs it through a prediction pipeline and returns a sentiment label, a confidence score, and a reason for the prediction. The reason tells you whether the call came from the ML model or from one of the rule-based checks (negation, sarcasm, mixed sentiment).

---

## How the prediction pipeline works

```
User Input
    │
    ▼
Emoji Sentiment Detection      ← "😭" signals negative before any text is read
    │
    ▼
Negation Rule Check            ← "not bad" → positive (runs before ML)
    │
    ▼
Sarcasm Detection              ← "yeah right", "oh great" → flip to negative
    │
    ▼
Mixed Sentiment Handling       ← focus shifts to the clause after "but" / "however"
    │
    ▼
Text Cleaning (preprocessing)  ← lowercase, strip URLs, expand slang, remove stopwords
    │
    ▼
TF-IDF Vectorization           ← top 5000 features, words → numbers
    │
    ▼
Logistic Regression Model      ← ML prediction
    │
    ▼
Confidence Threshold Check     ← below 45% → Neutral
    │
    ▼
Output: Sentiment + Confidence + Reason
```

The rule-based checks run *before* the ML model. If any rule fires with enough confidence, it short-circuits and the model is skipped. That is what the "Reason" field in the output is tracking.

---

## Project structure

```
sentiment-analysis/
│
├── app.py                  # Streamlit web app — entry point
├── model.py                # Prediction logic: rules + ML
├── preprocessing.py        # Text cleaning pipeline
├── feature_extraction.py   # TF-IDF vectorizer (top 5000 features)
├── train.py                # Training pipeline — run once
├── evaluate.py             # Accuracy, F1, confusion matrix
├── utils.py                # Dataset loading, model save/load with joblib
│
├── models/
│   ├── logistic_regression.joblib   # Pre-trained model, ready to use
│   └── tfidf_vectorizer.joblib      # Pre-trained vectorizer
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── emoji_dataset.csv
│   └── testdata.manual.2009.06.14.csv
│
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Size | Source |
|---|---|---|
| train.csv | Custom curated | Mixed reviews |
| test.csv | Custom curated | Mixed reviews |
| emoji_dataset.csv | Custom | Emoji-heavy tweets |
| Sentiment140 | 1.6M → sampled 50k | Twitter (Stanford) |
| **Total after sampling** | **~70,000+** | Combined |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords')"
```

**Run the app:**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Retrain from scratch (optional):**

The pre-trained model files are already in `models/`. Only run this if you want to retrain. Note: requires the full Sentiment140 file (`training.1600000.processed.noemoticon.csv`, 145 MB) which is not in the repo.

```bash
python train.py
```

---

## Model performance

Evaluated on a 20% stratified hold-out split from the training data.

| Metric | Score |
|---|---|
| Accuracy | ~79–82% |
| Precision | ~79–82% |
| Recall | ~79–82% |
| F1 Score | ~79–82% |

---

## Example predictions

| Input | Prediction | Confidence | Reason |
|---|---|---|---|
| "I love this product, it is amazing!" | POSITIVE | 99% | ML Prediction |
| "This is terrible, I hate it!" | NEGATIVE | 92% | ML Prediction |
| "It is okay, nothing special." | NEUTRAL | 45% | Low confidence → Neutral |
| "Not bad at all!" | POSITIVE | 100% | Negation Rule |
| "Yeah right, so amazing..." | NEGATIVE | 100% | Sarcasm detected |
| "The music was good but the story was boring" | NEGATIVE | ~70% | Mixed sentiment (after "but") |

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Algorithm | Logistic Regression (sklearn) |
| Feature Extraction | TF-IDF Vectorizer |
| Text Processing | NLTK, Regex |
| Web Framework | Streamlit |
| Model Serialization | Joblib |
| Data Handling | Pandas, NumPy |

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a pull request

---

## Acknowledgements

- [Sentiment140](http://help.sentiment140.com/) — 1.6M tweets, Stanford
- [NLTK](https://www.nltk.org/) — Natural Language Toolkit
- [Scikit-Learn](https://scikit-learn.org/) — ML library
- [Streamlit](https://streamlit.io/) — web app framework

---

