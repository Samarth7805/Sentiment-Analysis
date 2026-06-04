from model import predict_sentiment

texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "It is okay, nothing special."
]

for text in texts:
    prediction, confidence, reason = predict_sentiment(text)
    print(f"Text: '{text}' | Pred: {prediction} | Conf: {confidence:.4f} | Reason: {reason}")
