import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = "dataset/BBC News Summary/AG_News/train.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df["Text"] = (
        df["Title"].astype(str) + " " +
        df["Description"].astype(str)
    )

    documents = df["Text"].tolist()
    categories = df["Class Index"].tolist()

    return documents, categories

def create_tfidf():

    documents, categories = load_dataset()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Save vectorizer for ML and Flask app
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nDataset Loaded Successfully")
    print("Total Documents :", len(documents))
    print("Vocabulary Size :", len(vectorizer.vocabulary_))
    print("TF-IDF Shape :", tfidf_matrix.shape)

    return vectorizer, tfidf_matrix, documents, categories

if __name__ == "__main__":
    create_tfidf()

    