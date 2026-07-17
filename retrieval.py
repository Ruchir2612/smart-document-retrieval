from sklearn.metrics.pairwise import cosine_similarity


def search(query,
           vectorizer,
           tfidf_matrix,
           documents,
           categories,
           predicted_category=None,
           top_n=10):

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in top_indices:

        if similarity_scores[index] <= 0:
            continue

        if predicted_category is not None:

            if categories[index] != predicted_category:
                continue

        results.append({

            "score": round(float(similarity_scores[index]), 3),

            "category": categories[index],

            "document": documents[index]

        })

        if len(results) == top_n:
            break

    return results