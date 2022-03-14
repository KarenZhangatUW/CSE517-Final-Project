# load data
data2 = pd.read_csv('train.csv')
dict = {'text': 'post',
        'target': 'class'}
data2.rename(columns=dict,
          inplace=True)

# clean the text
post_ls = clean_corpus(data2)
datanew = pd.DataFrame({'post':post_ls,'class':data2['class']})

# split data into train, validation, and test sets
X2 = datanew['post']
y2 = datanew['class']
X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X2, y2, train_size=0.80, test_size=0.20, random_state=101)

X_train2.to_csv('x_train2.csv')
y_train2.to_csv('y_train2.csv')
X_test2.to_csv('x_test2.csv')
y_test2.to_csv('y_test2.csv')

# run the benchmark models listed in the paper
bm2 = BenchMark(x_tr=X_train2, x_te=X_test2, y_tr=y_train2,y_te=y_test2)
bm2.svm_unigram()
# LinearSVC(C=0.1)
# Test accuracy is 0.7892317793827971
# Test F1 is 0.7873634585438561
bm2.svm_tfidf()
# LinearSVC(C=0.1)
# Test accuracy is 0.7971109652002626
# Test F1 is 0.7932700036754898
bm2.svm_glove()
# LinearSVC(C=1))])
# Test accuracy is 0.6946815495732108
# Test F1 is 0.6925956780378557