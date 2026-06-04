# 🎭 Sentiment Analyzer

A **machine learning-powered** Sentiment Analysis web app that classifies text as **Positive**, **Negative**, or **Neutral** — built with Logistic Regression, TF-IDF, and Streamlit.

> **Live Demo:** [Deploy it yourself — instructions below!](#-deployment)

---

## 📸 Preview

> Enter any text, tweet, or review — and get instant sentiment analysis with confidence scores and reasoning!

---

## ✨ Features

- 🤖 **ML Model** — Logistic Regression trained on 70,000+ samples from multiple datasets
- 🧹 **Smart Preprocessing** — Handles slang, URLs, hashtags, mentions, HTML entities
- 😄 **Emoji Sentiment Detection** — Recognizes positive/negative emoji signals
- ❌ **Negation Handling** — Correctly classifies "not bad" as positive, "don't like" as negative
- 🙄 **Sarcasm Detection** — Catches patterns like "yeah right", "oh great", "wow thanks..."
- 🔀 **Mixed Sentiment** — Focuses on the clause after "but" / "however" / "although"
- 💡 **Explainable AI (XAI)** — Every prediction shows a reason (ML / Rule-based)
- 📊 **Confidence Score** — Displays the model's confidence level for each prediction
- 🌐 **Interactive Web UI** — Clean Streamlit interface, no coding needed to use

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── app.py                  # 🌐 Streamlit web application (main entry point)
├── model.py                # 🧠 Prediction logic — negation, sarcasm, mixed, ML
├── preprocessing.py        # 🧹 Text cleaning — slang, URLs, stopwords, emojis
├── feature_extraction.py   # 🔢 TF-IDF vectorizer — text → numbers
├── train.py                # 🏋️ Training pipeline — run this once to train
├── evaluate.py             # 📊 Model evaluation — accuracy, F1, confusion matrix
├── utils.py                # 🛠️ Helpers — load dataset, save/load model
│
├── models/
│   ├── logistic_regression.joblib  # ✅ Trained model (pre-trained, ready to use)
│   └── tfidf_vectorizer.joblib     # ✅ Trained vectorizer
│
├── data/
│   ├── train.csv                   # Training data
│   ├── test.csv                    # Test data
│   ├── emoji_dataset.csv           # Custom emoji-rich dataset
│   └── testdata.manual.2009.06.14.csv
│
├── requirements.txt        # 📦 Python dependencies
├── .gitignore              # 🚫 Files excluded from Git
└── README.md               # 📖 You are here!
```

---

## 🧠 How It Works

### Prediction Pipeline

```
User Input Text
       │
       ▼
① Emoji Sentiment Detection     ← "😭" → probably negative
       │
       ▼
② Negation Rule Check           ← "not bad" → positive (highest priority)
       │
       ▼
③ Sarcasm Detection             ← "yeah right" → flip to negative
       │
       ▼
④ Mixed Sentiment Handling      ← focus on part after "but"/"however"
       │
       ▼
⑤ Text Cleaning (preprocessing) ← lowercase, remove URLs, fix slang
       │
       ▼
⑥ TF-IDF Vectorization          ← convert words → numbers
       │
       ▼
⑦ Logistic Regression Model     ← ML prediction
       │
       ▼
⑧ Confidence Threshold Check    ← < 45% confidence → Neutral
       │
       ▼
  Final Output: Sentiment + Confidence + Reason
```

### Datasets Used

| Dataset | Size | Source |
|---|---|---|
| `train.csv` | Custom curated | Mixed reviews |
| `test.csv` | Custom curated | Mixed reviews |
| `emoji_dataset.csv` | Custom | Emoji-heavy tweets |
| `Sentiment140` | 1.6M (sampled 50k) | Twitter tweets |
| **Total after sampling** | **~70,000+** | Combined |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download NLTK stopwords (one-time)
python -c "import nltk; nltk.download('stopwords')"
```

### Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### (Optional) Retrain the Model

> ⚠️ The pre-trained model is already included. Only run this if you want to retrain from scratch.
> Note: Requires the large `training.1600000.processed.noemoticon.csv` file (145 MB, not in repo).

```bash
python train.py
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~79–82% |
| **Precision** | ~79–82% |
| **Recall** | ~79–82% |
| **F1 Score** | ~79–82% |

> *Evaluated on a 20% stratified hold-out test split.*

---

## 🧪 Example Predictions

| Input Text | Prediction | Confidence | Reason |
|---|---|---|---|
| "I love this product, it is amazing!" | ✅ POSITIVE | 99% | ML Prediction |
| "This is terrible, I hate it!" | ❌ NEGATIVE | 92% | ML Prediction |
| "It is okay, nothing special." | 😐 NEUTRAL | 45% | Low confidence → Neutral |
| "Not bad at all!" | ✅ POSITIVE | 100% | Negation Rule |
| "Yeah right, so amazing..." | ❌ NEGATIVE | 100% | Sarcasm detected |
| "The music was good but the story was boring" | ❌ NEGATIVE | ~70% | Mixed sentiment (after "but") |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML Algorithm** | Logistic Regression (sklearn) |
| **Feature Extraction** | TF-IDF Vectorizer |
| **Text Processing** | NLTK, Regex |
| **Web Framework** | Streamlit |
| **Model Serialization** | Joblib |
| **Data Handling** | Pandas, NumPy |

---

## 📁 Key Files Explained

| File | Role |
|---|---|
| `utils.py` | Loads datasets, saves/loads model files using joblib |
| `preprocessing.py` | Cleans raw text: lowercase → slang → URLs → stopwords |
| `feature_extraction.py` | Builds TF-IDF matrix (top 5000 features) |
| `model.py` | Core brain: rules (negation, sarcasm, mixed) + ML prediction |
| `train.py` | Full pipeline: load → clean → vectorize → train → evaluate → save |
| `evaluate.py` | Prints accuracy, precision, recall, F1 score, confusion matrix |
| `app.py` | Streamlit UI: takes user input, calls predict_sentiment(), shows result |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## 🙏 Acknowledgements

- [Sentiment140 Dataset](http://help.sentiment140.com/) — 1.6M tweets by Stanford
- [NLTK](https://www.nltk.org/) — Natural Language Toolkit
- [Scikit-Learn](https://scikit-learn.org/) — ML library
- [Streamlit](https://streamlit.io/) — Web app framework

---

<p align="center">Made with ❤️ and Python</p>
