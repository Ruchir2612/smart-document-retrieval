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


def expand_query(text):

    text = text.lower()

    expansions = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "cv": "computer vision",
        "nlp": "natural language processing",
        "ipl": "indian premier league",
        "wc": "world cup",
        "gdp": "gross domestic product",
        "covid": "coronavirus",
        "usa": "united states",
        "uk": "united kingdom"
    }

    words = text.split()

    expanded = []

    for word in words:
        if word in expansions:
            expanded.append(expansions[word])
        else:
            expanded.append(word)

    return " ".join(expanded)


def predict_category(text):

    text = expand_query(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    if confidence < 60:
        return (
            "Uncertain",
            round(confidence, 2)
        )

    return (
        category_names[prediction],
        round(confidence, 2)
    )