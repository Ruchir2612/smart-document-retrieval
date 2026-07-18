import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from tfidf_search import create_tfidf

CLASSIFIER_PATH = "models/classifier.pkl"


def train_model():

    # Create TF-IDF
    vectorizer, tfidf_matrix, documents, categories = create_tfidf()

    X = tfidf_matrix
    y = categories

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Trained Successfully")
    print("\nAccuracy :", round(accuracy * 100, 2), "%")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    joblib.dump(model, CLASSIFIER_PATH)

    print("\nModel Saved Successfully")

    return model


if __name__ == "__main__":
    train_model()