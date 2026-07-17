import joblib
from tfidf_search import create_tfidf

print("Loading dataset and creating TF-IDF...")

vectorizer, tfidf_matrix, documents, categories = create_tfidf()

joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")
joblib.dump(documents, "models/documents.pkl")
joblib.dump(categories, "models/categories.pkl")

print("\nAll TF-IDF files saved successfully!")