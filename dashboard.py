import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/BBC News Summary/AG_News/train.csv"

df = pd.read_csv(DATASET_PATH)

category_names = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

counts = df["Class Index"].value_counts().sort_index()

labels = [category_names[i] for i in counts.index]
values = counts.values

# -----------------------------
# Bar Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.bar(labels, values)

plt.title("Dataset Distribution")

plt.xlabel("Category")

plt.ylabel("Documents")

plt.tight_layout()

plt.savefig("static/category_distribution.png")

plt.close()

# -----------------------------
# Pie Chart
# -----------------------------

plt.figure(figsize=(6,6))

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Category Percentage")

plt.tight_layout()

plt.savefig("static/category_pie.png")

plt.close()

print("Dashboard Graphs Created Successfully")