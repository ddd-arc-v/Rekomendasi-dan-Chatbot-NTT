import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('Datasetwisatabot.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# # Mengakses elemen dalam struktur JSON
# aksesjson = intents['intents']  # Mengakses daftar intents
# intent_index = 0 

# # Menampilkan hanya "tag" dan "patterns"
# for i in intents:
#     print("Tag:", intents[intent_index]['tag'])
#     print("Patterns:", intents[intent_index]['patterns'][0])  
#     print("Response:", intents[intent_index]['responses'][0]) 

print(len(documents), "dokumen")
print(len(classes), "kelas", classes)
print(len(words), "kata unik yang sudah dilakukan lemmatisasi", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inisialisasi data latih
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Padding urutan (sequences) agar panjangnya sama
max_sequence_length = max(len(x[0]) for x in training)
for i in range(len(training)):
    training[i][0] = pad_sequences([training[i][0]], maxlen=max_sequence_length)[0]

# Pisahkan bag of words dan output row sebelum dikonversi menjadi NumPy array
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Pad vektor fitur agar memiliki panjang yang sama
max_sequence_length = max(len(x) for x in train_x)
train_x = pad_sequences(train_x, maxlen=max_sequence_length)

print("Training data created")

# Membangun model RNN
model = Sequential()
model.add(Embedding(len(words), 128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model dan menyimpannya
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_rnn_model.h5')

print("Model RNN telah dibuat")
