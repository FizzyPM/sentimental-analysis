from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Activation, GlobalMaxPooling1D, GRU
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import numpy as np
# import random
# import gensim
# from scipy.sparse import hstack
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm import tqdm
# from sklearn.metrics import f1_score
# from gensim.models.word2vec import Word2Vec
# from sklearn.neighbors.nearest_centroid import NearestCentroid
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from numpy import array

data_labels = []
data2 = []
df = open("preprocessed-data.txt", "r")
for i in df:
    cols = i.split("\t")
    # data_labels.append(cols[0])
    if cols[0] == 'positive':
        data_labels.append(1)
        data2.append(cols[1])
    elif cols[0] == 'negative':
        data_labels.append(0)
        data2.append(cols[1])
    else:
        pass
    # print(cols[0])
df.close()

corpus = data2


x_train, x_val, y_train, y_val = train_test_split(corpus, data_labels, test_size=0.20, train_size=0.80, random_state=1234)

# ------------------------------------GLOVE classifier---------------------------------------

# embeddings
embeddings_index = dict()
f = open("glove.6B.300d.txt", encoding="utf8")
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
t = Tokenizer()
t.fit_on_texts(x_train)
encoded_docs = t.texts_to_sequences(x_train) 
# print(encoded_docs[:5])
vocabulary_size = len(t.word_index) + 1
# encoded_docs = encoded_docs[8678:]
# print(len(encoded_docs))
# print(len(data_labels))
# print(len(data2))

for x in x_train[:5]:
    print(x)
for x in encoded_docs[:5]:
    print(x)

# length = []
# for x in x_train:
#     length.append(len(x.split()))
# print(max(length))

padded_docs = pad_sequences(encoded_docs, maxlen=40)
print(padded_docs[:5])

encoded_docs_val = t.texts_to_sequences(x_val)
padded_docs_val = pad_sequences(encoded_docs_val, maxlen=40)
# print(x_val_seq[:5])

num_words = vocabulary_size
embedding_matrix = np.zeros((num_words, 300))
for word, i in t.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# print(np.array_equal(embedding_matrix[24], embeddings_index.get('going')))

# print(len(data_labels))
# print(len(corpus))
# print(len(padded_docs))

#  -------------------------------CNN-------------------------------------------------------

print("--------------------------------CNN------------------------------")
model_glove = Sequential()
e = Embedding(num_words, 300, weights=[embedding_matrix], input_length=40, trainable=True)
model_glove.add(e)
model_glove.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_glove.add(Flatten())
# model_glove.add(Dense(units=256, activation='relu'))
model_glove.add(Dense(units=1, activation='sigmoid'))
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_glove.fit(padded_docs, np.array(y_train), validation_data=(padded_docs_val, np.array(y_val)), batch_size=64, epochs=8, verbose=2)

# print(model_glove.summary())
# print(data_labels)

# ---------------------------------lstm---------------------------------------------------------------- 
# print(vocabulary_size)
# print(num_words)
print("--------------------------------LSTM------------------------------")
model = Sequential()
e = Embedding(num_words, 300, weights=[embedding_matrix], input_length=40, trainable=True)
model.add(e)
# model.add(Dropout(0.2))
model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(units=256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_docs, np.array(y_train), validation_data=(padded_docs_val, np.array(y_val)), batch_size=64, epochs=8, verbose=2)

# -----------------------------------cnn+lstm----------------------------------------------------------

print("--------------------------------CNN+LSTM------------------------------")
model_hy = Sequential()
model_hy.add(Embedding(vocabulary_size, 300, input_length=40, weights=[embedding_matrix], trainable=True))
model_hy.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_hy.add(MaxPooling1D(pool_size=2))
model_hy.add(LSTM(100))
# model_hy.add(Dense(units=256, activation='relu'))
model_hy.add(Dense(units=1, activation='sigmoid'))
model_hy.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_hy.fit(padded_docs, np.array(y_train), validation_data=(padded_docs_val, np.array(y_val)), batch_size=64, epochs=8, verbose=2)
