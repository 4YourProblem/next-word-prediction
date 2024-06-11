import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np

file = open("data.txt", "r", encoding="utf8")
# store file in list
lines = []
for i in file:
    lines.append(i)
# convert list to string
data = ""
for i in lines:
    data = " ".join(lines)
# relace with space
data = (
    data.replace("\n", "")
    .replace("\r", "")
    .replace("\ufeff", "")
    .replace("“", "")
    .replace("”", "")
)

# remove spaces
data = data.split()
data = " ".join(data)
# print(data[:1000])

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# save the tokenizer for presict function
pickle.dump(tokenizer, open("token1.pkl", "wb"))
sequence_data = tokenizer.texts_to_sequences([data])[0]
print(sequence_data[:15])
# len(sequence_data)

vocab_size = len(tokenizer.word_index) + 1

sequence = []
for i in range(3, len(sequence_data)):
    words = sequence_data[i - 3 : i + 1]
    sequence.append(words)
print("Lenght sequence :", len(sequence))
sequence = np.array(sequence)
X = []
y = []

for i in sequence:
    X.append(i[0:3])
    y.append(i[3])
X = np.array(X)
y = np.array(y)
print("Data:", X[:10])
print("Response:", y[:10])
y = to_categorical(y, num_classes=vocab_size)
y[:5]


# Create the model
def create_model():
    embedding_dim = 128  # Embedding dimension
    lstm_units = 256  # Number of LSTM units
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=sequence))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(X, y, epochs=5, batch_size=64)
    model.save("lstm_model.keras")
    return


def predict_top_words(text, num_predictions=5):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    model = tf.keras.models.load_model("lstm_model.keras")
    predictions = model.predict(sequence)
    top_indices = predictions.argsort(axis=1)[:, -num_predictions:]

    top_words = []
    for index_list in top_indices.T:
        word_list = []
        for index in index_list:
            predicted_word = tokenizer.index_word[index]
            word_list.append(predicted_word)
        top_words.append(word_list)

    return top_words
