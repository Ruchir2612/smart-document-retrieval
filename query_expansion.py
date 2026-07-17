import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def expand_query(query):

    expanded=[]

    for word in query.split():

        expanded.append(word)

        synsets = wordnet.synsets(word)

        # only first synonym
        if synsets:

            lemmas = synsets[0].lemmas()

            if len(lemmas) > 0:

                expanded.append(
                    lemmas[0].name()
                )

    return " ".join(
        set(expanded)
    )