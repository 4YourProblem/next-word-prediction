from n_gram import predict_next_word as n_gram_predict
from lstm import predict_top_words as lstm_gram_predict
import itertools

def get_all_predictions(text_sentence, top_clean=5):
    # =========== N-Gram ===========
    n_gram_predicted_word = []
    text_sentence = text_sentence.lower().split()
    print(len(text_sentence))
    print(text_sentence)
    prediction = n_gram_predict(tuple(text_sentence), len(text_sentence) + 1)
    print(prediction)
    for ngram, freq in prediction:
        n_gram_predicted_word.append(ngram[-1])
        print(ngram)
        print(freq)
    print(n_gram_predicted_word)
    # =========== LSTM ===========
    lstm_predicted_word = []
    # text_sentence = text_sentence.lower().split()
    top_predictions = lstm_gram_predict(text_sentence)
    # print(top_predictions)
    for i, prediction_list in enumerate(top_predictions):
        lstm_predicted_word.append(prediction_list)
    print(lstm_predicted_word)
    flat_list = list(itertools.chain.from_iterable(lstm_predicted_word))
    print(flat_list)
    return {"ngram": "\n".join(n_gram_predicted_word),
            "lstm":"\n".join(flat_list)}


