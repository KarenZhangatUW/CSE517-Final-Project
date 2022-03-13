import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator

import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# load the train and test sets we have processed
X_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')



# LinearSVC + unigram
SVCn_pipeline = Pipeline([
                ('ngram', CountVectorizer(analyzer='word', ngram_range=(1, 1))),
                ('clf', LinearSVC()),
            ])
grid_param = [{'clf__C': [0.1, 1, 10, 100, 1000]}]
gridsearch = GridSearchCV(SVCn_pipeline, grid_param, cv=4)
best_SVCn = gridsearch.fit(X_train, y_train)
best_SVCn_pred = gridsearch.predict(X_test)
print(best_SVCn.best_estimator_)
print('Test accuracy is {}'.format(accuracy_score(y_test, best_SVCn_pred)))
print('Test F1 is {}'.format(f1_score(y_test, best_SVCn_pred, average='weighted')))


# GloVe + SVM
class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        # Doc.vector defaults to an average of the token vectors.
        # https://spacy.io/api/doc#vector
        return [self.nlp(text).vector for text in X]


embeddings_pipeline = Pipeline(
    steps=[
        ("mean_embeddings", SpacyVectorTransformer(nlp)),
        ('clf', LinearSVC())
    ]
)

grid_param = [{'clf__C': [0.1, 1, 10, 100, 1000]}]
gridsearch = GridSearchCV(embeddings_pipeline, grid_param, cv=4)
best_embedding = gridsearch.fit(X_train, y_train)
best_embedding_pred = gridsearch.predict(X_test)
print(best_embedding.best_estimator_)
print('Test accuracy is {}'.format(accuracy_score(y_test, best_embedding_pred)))
print('Test F1 is {}'.format(f1_score(y_test, best_embedding_pred, average='weighted')))


# LinearSVC + TF-IDF
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', LinearSVC()),
            ])
grid_param = [{'clf__C': [0.1, 1, 10, 100, 1000]}]
gridsearch = GridSearchCV(SVC_pipeline, grid_param, cv=4)
best_SVC = gridsearch.fit(X_train, y_train)
best_SVC_pred = gridsearch.predict(X_test)
print(best_SVC.best_estimator_)
print('Test accuracy is {}'.format(accuracy_score(y_test, best_SVC_pred)))
print('Test F1 is {}'.format(f1_score(y_test, best_SVC_pred, average='weighted')))


# Naive Bayes + TF-IDF
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
NB_pipeline.fit(X_train, y_train)
y_pred = NB_pipeline.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))
print('Test F1 is {}'.format(f1_score(y_test, y_pred, average='weighted')))


# Logistic Regression + TF-IDF
LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
LogReg_pipeline.fit(X_train, y_train)
y_pred3 = LogReg_pipeline.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred3)))
print('Test F1 is {}'.format(f1_score(y_test, y_pred3, average='weighted')))