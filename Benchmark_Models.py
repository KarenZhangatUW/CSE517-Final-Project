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


# transformer for glove
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


class BenchMark:
    def __init__(self, x_tr, x_te, y_tr, y_te):
        self.X_tr = x_tr
        self.X_te = x_te
        self.y_tr = y_tr
        self.y_te = y_te

    def svm_unigram(self): # LinearSVC + unigram
        print('SVM(unigram): \n')
        SVCn_pipeline = Pipeline([
            ('ngram', CountVectorizer(analyzer='word', ngram_range=(1, 1))),
            ('clf', LinearSVC()),
        ])
        grid_param = [{'clf__C': [0.1, 1, 10, 100, 1000]}]
        gridsearch = GridSearchCV(SVCn_pipeline, grid_param, cv=4)
        best_SVCn = gridsearch.fit(self.X_tr, self.y_tr)
        best_SVCn_pred = gridsearch.predict(self.X_te)
        print(best_SVCn.best_estimator_)
        print('Test accuracy is {}'.format(accuracy_score(self.y_te, best_SVCn_pred)))
        print('Test F1 is {}'.format(f1_score(self.y_te, best_SVCn_pred, average='weighted')))

    def svm_glove(self):
        # GloVe + SVM
        print('SVM(GloVe): \n')
        embeddings_pipeline = Pipeline(
            steps=[
                ("mean_embeddings", SpacyVectorTransformer(nlp)),
                ('clf', LinearSVC())
            ]
        )

        grid_param = [{'clf__C': [0.1, 1, 10, 100, 1000]}]
        gridsearch = GridSearchCV(embeddings_pipeline, grid_param, cv=4)
        best_embedding = gridsearch.fit(self.X_tr, self.y_tr)
        best_embedding_pred = gridsearch.predict(self.X_te)
        print(best_embedding.best_estimator_)
        print('Test accuracy is {}'.format(accuracy_score(self.y_te, best_embedding_pred)))
        print('Test F1 is {}'.format(f1_score(self.y_te, best_embedding_pred, average='weighted')))

    def svm_tfidf(self):
        # LinearSVC + TF-IDF
        print('SVM(tfidf): \n')
        SVC_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', LinearSVC()),
        ])
        grid_param = [{'clf__C': [0.1, 1, 10, 100, 1000]}]
        gridsearch = GridSearchCV(SVC_pipeline, grid_param, cv=4)
        best_SVC = gridsearch.fit(self.X_tr, self.y_tr)
        best_SVC_pred = gridsearch.predict(self.X_te)
        print(best_SVC.best_estimator_)
        print('Test accuracy is {}'.format(accuracy_score(self.y_te, best_SVC_pred)))
        print('Test F1 is {}'.format(f1_score(self.y_te, best_SVC_pred, average='weighted')))

    def nb_tfidf(self):
        print('NB(tfidf): \n')
        # Naive Bayes + TF-IDF
        NB_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])
        NB_pipeline.fit(self.X_tr, self.y_tr) # train
        y_pred = NB_pipeline.predict(self.X_te) # test
        print('Test accuracy is {}'.format(accuracy_score(self.y_te, y_pred)))
        print('Test F1 is {}'.format(f1_score(self.y_te, y_pred, average='weighted')))

    def lr_tfidf(self):
        print('lr(tfidf): \n')
        # Logistic Regression + TF-IDF
        LogReg_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
        ])
        LogReg_pipeline.fit(self.X_tr, self.y_tr)
        y_pred3 = LogReg_pipeline.predict(self.X_te)
        print('Test accuracy is {}'.format(accuracy_score(self.y_te, y_pred3)))
        print('Test F1 is {}'.format(f1_score(self.y_te, y_pred3, average='weighted')))


# load the train and test sets we have processed
X_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')

# run the benchmark models listed in the paper
bm = BenchMark(x_tr=X_train, x_te=X_test, y_tr=y_train,y_te=y_test)
bm.svm_unigram()
bm.svm_tfidf()
bm.svm_glove()

# extra models
bm.lr_tfidf()
bm.nb_tfidf()

# run for another data set
X_train2 = pd.read_csv('x_train2.csv')
y_train2 = pd.read_csv('y_train2.csv')
X_test2 = pd.read_csv('x_test2.csv')
y_test2 = pd.read_csv('y_test2.csv')
bm = BenchMark(x_tr=X_train, x_te=X_test, y_tr=y_train,y_te=y_test)
bm.svm_unigram()
bm.svm_tfidf()
bm.svm_glove()

