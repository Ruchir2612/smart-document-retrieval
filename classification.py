import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from tfidf_search import create_tfidf

CLASSIFIER_PATH = "models/classifier.pkl"

def train_model():

    # Use the SAME TF-IDF vectorizer as retrieval
    vectorizer, tfidf_matrix, documents, categories = create_tfidf()

    X = tfidf_matrix
    y = categories

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Trained Successfully")
    print("\nAccuracy :", round(accuracy * 100, 2), "%")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    # Save trained model
    joblib.dump(model, CLASSIFIER_PATH)

    print("\nModel Saved Successfully")

    return model, vectorizer

def predict_category(text):

    model = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load("models/vectorizer.pkl")

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    category_names = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech"
    }

    return category_names[prediction]

if __name__ == "__main__":
    train_model()