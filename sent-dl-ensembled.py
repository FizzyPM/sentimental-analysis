from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Activation, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np
# import random
# import gensim
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
t.fit_on_texts(corpus)
encoded_docs = t.texts_to_sequences(corpus)  # encode words to integers
# print(encoded_docs[:5])
vocabulary_size = len(t.word_index) + 1
# print(vocabulary_size)
# encoded_docs = encoded_docs[8678:]
# print(len(encoded_docs))
# print(len(data_labels))
# print(len(data2))

for x in corpus[:5]:
    print(x)
for x in encoded_docs[:5]:
    print(x)

# length = []
# for x in x_train:
#     length.append(len(x.split()))
# print(max(length))

padded_docs = pad_sequences(encoded_docs, maxlen=30)
print(padded_docs[:5])

num_words = vocabulary_size
embedding_matrix = np.zeros((num_words, 300))
for word, i in t.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(np.array_equal(embedding_matrix[24], embeddings_index.get('going')))

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
hybrid = hstack([bow, tfidf, padded_docs])
hy = hybrid.tocsr()

x_train, x_test, y_train, y_test = train_test_split(hy, data_labels, test_size=0.10, train_size=0.90, random_state=1234)

# print(bow.get_shape())
# print(tfidf.get_shape())
# print(len(xpadded_docs))
# print(len(data_labels))
# print(hybrid.get_shape())
# print(xvocabulary_size)

print("--------------------------------CNN on Ensembled Features------------------------------")
model_glove = Sequential()
e = Embedding(vocabulary_size, 300, weights=[embedding_matrix], input_length=25884, trainable=True)
model_glove.add(e)
model_glove.add(Flatten())
# model_glove.add(Dense(units=256, activation='relu'))
model_glove.add(Dense(units=1, activation='sigmoid'))
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_glove.fit(x_train, np.array(y_train), validation_data=(x_test, np.array(y_test)), batch_size=64, epochs=25, verbose=2)


print("--------------------------------LSTM on Ensembled Features------------------------------")
model = Sequential()
e = Embedding(vocabulary_size, 300, weights=[embedding_matrix], input_length=25884, trainable=True)
model.add(e)
# model.add(Dropout(0.2))
model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(units=256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, np.array(y_train), validation_data=(x_test, np.array(y_test)), batch_size=64, epochs=25, verbose=2)
