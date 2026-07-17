from predict import predict_category

while True:

    query = input("Enter Text : ")

    if query == "exit":
        break

    category, confidence = predict_category(query)

    print()

    print("Predicted Category :", category)

    print("Confidence :", confidence, "%")

    print()