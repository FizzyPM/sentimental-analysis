from sklearn import tree
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.metrics import classification_report
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Flatten # , LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.layers.embeddings import Embedding
# import random
# import gensim
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import f1_score
# from gensim.models.word2vec import Word2Vec
# from sklearn.neighbors.nearest_centroid import NearestCentroid


data_labels = []
# for line in f:
#     cols = line.split("\t")
#     data1.append(cols[5])
# f.close()
data2 = []
data1 = []
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

df = open("preprocessed-data.txt", "r")
for i in df:
    cols = i.split("\t")
    # data_labels.append(cols[0])
    if cols[0] == 'positive':
        data_labels.append(1)
        data1.append(cols[1])
    elif cols[0] == 'negative':
        data_labels.append(0)
        data1.append(cols[1])
    else:
        pass
    # print(cols[0])
df.close()

corpus = data2 + data1

# -----------------------------------LOGISTIC RERGRESSION-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
# print(bow.toarray())
# print(bow_vectorizer.get_feature_names())
# print(bow.get_shape())
# print(len(data_labels))
X_train, X_test, y_train, y_test = train_test_split(bow, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________LOGISTIC RERGRESSION________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
# print(tfidf)
# print(tfidf.get_shape())
X_train, X_test, y_train, y_test = train_test_split(tfidf, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()  # Convert this matrix to Compressed Sparse Row format
X_train, X_test, y_train, y_test = train_test_split(hy, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))

# ---------------------------------PASSIVE AGGRESSIVE CLASSIFIER--------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____PASSIVE AGGRESSIVE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


# -------------------------------------MULTINOMIAL NB-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________MULTINOMIAL NB__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data_labels, train_size=0.70, test_size=0.30, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


# -------------------------------------PERCEPTRON-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________PERCEPTRON__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))

# -------------------------------------RANDOM FOREST CLASSIFIER-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf = rf.fit(X=X_train, y=y_train)
y_pred = rf.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________RANDOM FOREST__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf = rf.fit(X=X_train, y=y_train)
y_pred = rf.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf = rf.fit(X=X_train, y=y_train)
y_pred = rf.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


# -------------------------------------LINEAR SVC-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(bow, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________LINEAR SVC__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data_labels, test_size=0.30, train_size=0.70, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy, data_labels, train_size=0.70, test_size=0.30, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))

'''
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
# print(bow.toarray())
# print(bow_vectorizer.get_feature_names())
# print(bow.get_shape())
# print(len(data_labels))
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________LOGISTIC RERGRESSION________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")
# print(classification_report(y_test, y_pred))
'''
