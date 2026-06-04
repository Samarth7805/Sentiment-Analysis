
from sklearn.model_selection import train_test_split
from utils import load_dataset, save_model
from preprocessing import preprocess_dataframe
from feature_extraction import build_tfidf, save_vectorizer
from model import get_model, MODEL_FILE
from evaluate import evaluate_model


def main():
    
    df = load_dataset()

    print(df["sentiment"].value_counts())

    df = preprocess_dataframe(df)

    X, vectorizer = build_tfidf(df["cleaned_text"])
    y = df["sentiment"]

    save_vectorizer(vectorizer)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")
    model = get_model()
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    save_model(model, MODEL_FILE)

    evaluate_model(model, X_test, y_test)

    

if __name__ == "__main__":
    main()
