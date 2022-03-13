import pandas as pd
import sklearn.model_selection as model_selection
import re
import matplotlib.pyplot as plt

# load data
dp = pd.read_csv("implicit_hate_v1_stg2_posts.tsv", delimiter='\t')


# clean text
def clean_sent(post):  # cleaning each sentence
    # convert to lower case
    s = post.lower()
    # remove punctuation
    s = re.sub(r'[^\w\s]','',post)
    return s


def clean_corpus(dt):  # cleaning the whole corpus
    post_ls = []
    for i in range(dt.shape[0]):
        post = dt['post'].iloc[i]
        post = clean_sent(post)
        post_ls.append(post)
    return post_ls


post_ls = clean_corpus(dp)
data = pd.DataFrame({'post':post_ls,'class':dp['implicit_class']})

# get statistics of classes
counts = []
categories = list(data['class'].unique())
for i in categories:
    counts.append((i, (data['class']==i).sum()))
dt_stats = pd.DataFrame(counts, columns=['category', 'number_of_posts'])
# plot the statistics
dt_stats.plot(x='category', y='number_of_posts', kind='bar', legend=False, grid=True, figsize=(7, 4))
plt.xticks(rotation=15)
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)


# split data into train, validation, and test sets
X = data['post']
y = data['class']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

X_train.to_csv('x_train.csv')
y_train.to_csv('y_train.csv')
X_test.to_csv('x_test.csv')
y_test.to_csv('y_test.csv')
