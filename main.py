from n_gram import predict_next_word as n_gram_predict


def get_all_predictions(text_sentence, top_clean=5):
    # =========== N-Gram ===========
    n_gram_predicted_word = []
    text_sentence = text_sentence.lower().split()
    print(len(text_sentence))
    print(text_sentence)
    prediction = n_gram_predict(tuple(text_sentence), len(text_sentence) + 1)

    for ngram, freq in prediction:
        n_gram_predicted_word.append(ngram[-1])

    return {"ngram": "\n".join(n_gram_predicted_word)}
