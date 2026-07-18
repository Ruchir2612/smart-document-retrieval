from flask import Flask, render_template, request, Response
from tfidf_search import create_tfidf
from retrieval import search
from predict import predict_category

import csv
import io
import time

app = Flask(__name__)

# Load Search Engine
vectorizer, tfidf_matrix, documents, categories = create_tfidf()

category_names = {
    1: "🌍 World",
    2: "⚽ Sports",
    3: "💼 Business",
    4: "💻 Sci/Tech"
}

category_to_index = {
    "World": 1,
    "Sports": 2,
    "Business": 3,
    "Sci/Tech": 4
}

search_history = []
latest_results = []


@app.route("/")
def landing():
    return render_template("home.html")


@app.route("/search", methods=["GET", "POST"])
def home():

    global latest_results

    results = []
    query = ""
    predicted_category = ""
    confidence = ""
    search_time = 0

    if request.method == "POST":

        start = time.time()

        query = request.form["query"]

        if query not in search_history:
            search_history.insert(0, query)

        if len(search_history) > 10:
            search_history.pop()

        predicted_category, confidence = predict_category(query)

        predicted_index = category_to_index.get(predicted_category)

        if predicted_index is None:

            results = search(
                query=query,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                documents=documents,
                categories=categories,
                top_n=10
            )

        else:

            results = search(
                query=query,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                documents=documents,
                categories=categories,
                predicted_category=predicted_index,
                top_n=10
            )

        for item in results:
            item["category"] = category_names.get(
                item["category"],
                "Unknown"
            )

        latest_results = results

        search_time = round(time.time() - start, 3)

    return render_template(
        "index.html",
        results=results,
        query=query,
        history=search_history,
        predicted_category=predicted_category,
        confidence=confidence,
        search_time=search_time,
        total_documents=len(documents)
    )


@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        total_documents=len(documents),
        vocabulary_size=len(vectorizer.vocabulary_),
        accuracy=91.07
    )


@app.route("/download")
def download():

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Category",
        "Similarity Score",
        "Document"
    ])

    for row in latest_results:

        writer.writerow([
            row["category"],
            row["score"],
            row["document"]
        ])

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=search_results.csv"
        }
    )


if __name__ == "__main__":
    app.run(debug=True)