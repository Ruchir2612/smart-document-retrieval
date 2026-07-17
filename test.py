from tfidf_search import create_tfidf
from retrieval import search

# Load dataset and create TF-IDF
vectorizer, tfidf_matrix, documents, categories = create_tfidf()

category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

search_history = []

print("=" * 60)
print("      Intelligent Document Retrieval System")
print("=" * 60)

while True:

    query = input("\nEnter Search Query (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    search_history.append(query)

    results = search(
        query,
        vectorizer,
        tfidf_matrix,
        documents,
        categories
    )

    if len(results) == 0:
        print("\nNo matching documents found.")
        continue

    print("\nTop Matching Documents\n")

    for i, result in enumerate(results, start=1):

        print("=" * 60)
        print(f"Result {i}")
        print("Category :", category_names.get(result["category"], "Unknown"))
        print("Similarity Score :", result["score"])
        print("-" * 60)

        print(result["document"][:500])
        print()

print("\nSearch History")

for item in search_history:
    print("-", item)