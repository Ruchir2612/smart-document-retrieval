import joblib

# Load trained model and vectorizer
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

    # Phrase-level expansions (checked first)
    phrase_expansions = {
        "artificial intelligence": "artificial intelligence machine learning deep learning neural network computer science technology ai",
        "machine learning": "machine learning artificial intelligence deep learning data science computer science",
        "deep learning": "deep learning machine learning artificial intelligence neural network",
        "neural network": "neural network deep learning artificial intelligence machine learning",
        "stock market": "stock market shares trading finance business economy investment",
        "world cup": "world cup football cricket fifa sports tournament",
        "indian premier league": "indian premier league ipl cricket sports",
        "computer vision": "computer vision image processing artificial intelligence machine learning",
        "natural language processing": "natural language processing nlp artificial intelligence machine learning"
    }

    if text in phrase_expansions:
        return phrase_expansions[text]

    # Single-word expansions
    word_expansions = {
        "ai": "artificial intelligence machine learning deep learning",
        "ml": "machine learning artificial intelligence",
        "cv": "computer vision image processing",
        "nlp": "natural language processing text mining",
        "ipl": "indian premier league cricket sports",
        "wc": "world cup football cricket sports",
        "gdp": "gross domestic product economy business",
        "covid": "coronavirus pandemic health",
        "usa": "united states america",
        "uk": "united kingdom britain",
        "stocks": "stock market shares finance business",
        "share": "stock market investment business",
        "finance": "business economy investment"
    }

    words = text.split()

    expanded = []

    for word in words:
        expanded.append(word_expansions.get(word, word))

    return " ".join(expanded)


def predict_category(text):

    expanded_text = expand_query(text)

    vector = vectorizer.transform([expanded_text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = float(round(max(probabilities) * 100, 2))

    return category_names[prediction], confidence