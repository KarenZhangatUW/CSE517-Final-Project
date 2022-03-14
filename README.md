# CSE517-Final-Project
Replication for the paper: Latent Hatred: A Benchmark for Understanding Implicit Hate Speech (2021)

## Data Download Instruction:

You could download the original data set through the following process: first complete a short survey through https://forms.gle/QxCpEbVp91Z35hWFA. Then follow this link to download: https://www.dropbox.com/s/24meryhqi1oo0xk/implicit-hate-corpus-nov-2021.zip?dl=0.

You could also download the data sets we processed based on the original data. In this repo, we offer x_train.csv, y_train.csv, x_test.csv, y_test.csv, which are posts and target (class) of training set and test set respectively.


## File Description:

preprocessing.py  

Python codes for cleaning and splitting the originl data set

Benchmark_Models.py  

Python codes for implementing the benchmark models for classification including SVM(unigram), SVM(TF-IDF), SVM(GloVe), Naive Bayesian(TF-IDF), and Logistic Regression(TF-IDF) 

Bert_based_implicit_hate_classif...  

Python (Google Colab) codes for implementing the BERT model for classification




