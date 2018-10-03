from sklearn import tree
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
# import random
# import gensim
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import f1_score
# from gensim.models.word2vec import Word2Vec
# from sklearn.neighbors.nearest_centroid import NearestCentroid

f = open("./preprocessed-data/preprocessed-test-A.txt", "r")
data1 = []
data_labels = []
for line in f:
    cols = line.split("\t")
    data1.append(cols[5])
f.close()
data2 = []
df = open("./preprocessed-data/preprocessed-train-A.txt", "r")
for i in df:
    cols = i.split("\t")
    data2.append(cols[1])
    data_labels.append(cols[0])
df.close()

corpus = data1 + data2

# -----------------------------------LOGISTIC RERGRESSION-----------------------------------
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


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
# print(tfidf)
# print(tfidf.get_shape())
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
# hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
# hy = hy_vectorizer.fit_transform(corpus)
hybrid = hstack([tfidf, bow])
hy = hybrid.tocsr()  # Convert this matrix to Compressed Sparse Row format
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

# ---------------------------------PASSIVE AGGRESSIVE CLASSIFIER--------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____PASSIVE AGGRESSIVE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------MULTINOMIAL NB-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________MULTINOMIAL NB__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------PERCEPTRON-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________PERCEPTRON__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -----------------------------------RIDGE CLASSIFIER------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________RIDGE CLASSIFIER__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------LINEAR SVC-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________LINEAR SVC__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -----------------------------------DECISION TREE CLASSIFIER-----------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(corpus)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_bow, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____DECISION TREE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, data_labels, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(corpus)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(train_hy, data_labels, train_size=0.80, test_size=0.20, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# print(y_pred)
# test_pred = log_model.predict(test_bow)  ### PREDICTING TEST DATA SET ###
# print(test_pred)

# x_train, x_test, y_train, y_test = train_test_split(data2, data_labels, train_size=0.80)
# tweet_w2v = Word2Vec(size=200, min_count=10)
# tweet_w2v.build_vocab([x.split() for x in tqdm(x_train)])
# tweet_w2v.train([x.split() for x in tqdm(x_train)])
