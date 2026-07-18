import joblib

model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}


def expand_query(text):

    text = text.lower().strip()

    phrase_expansions = {

        "artificial intelligence":
        "artificial intelligence machine learning deep learning computer science technology",

        "machine learning":
        "machine learning artificial intelligence deep learning",

        "deep learning":
        "deep learning machine learning artificial intelligence",

        "computer vision":
        "computer vision image processing artificial intelligence",

        "natural language processing":
        "natural language processing artificial intelligence",

        "stock market":
        "stock market shares finance business economy",

        "world cup":
        "world cup football cricket sports",

        "indian premier league":
        "indian premier league cricket sports"

    }

    if text in phrase_expansions:
        return phrase_expansions[text]

    word_expansions = {

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
        expanded.append(word_expansions.get(word, word))

    return " ".join(expanded)


def predict_category(text):

    text = expand_query(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = round(max(probabilities) * 100, 2)

    return category_names[prediction], confidence