import joblib

# Load saved model
model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}


def predict_category(text):

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    return (
        category_names[prediction],
        round(confidence, 2)
    )