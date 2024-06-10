# Import Necessary Libraries

import nltk
from nltk import ngrams
from collections import defaultdict
import random


f = open("data.txt", "r", encoding="utf-8")
text = f.read()


# Define Function
def predict_next_word(prefix, ngram_order=3):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Preprocess the words (convert to lowercase, remove punctuation)
    words = [word.lower() for word in words if word.isalnum()]

    # Define the order of the N-gram model (N=3 for trigrams)
    N = 6

    # Create N-grams from the tokenized words
    ngrams_list = list(ngrams(words, ngram_order))

    # Create a defaultdict to store N-grams and their frequency
    ngram_freq = defaultdict(int)
    for ngram in ngrams_list:
        ngram_freq[ngram] += 1

    # Filter N-grams that start with the given prefix
    matching_ngrams = [
        (ngram, freq) for ngram, freq in ngram_freq.items() if ngram[:-1] == prefix
    ]

    if not matching_ngrams:
        return "No prediction available."

    # Sort N-grams by frequency in descending order
    sorted_ngrams = sorted(matching_ngrams, key=lambda x: x[1], reverse=True)

    # Select the N-gram with the highest frequency as the prediction
    prediction = sorted_ngrams[:5]

    return prediction
