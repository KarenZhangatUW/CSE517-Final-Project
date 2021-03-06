# CSE517-Final-Project
Replication for the paper: Latent Hatred: A Benchmark for Understanding Implicit Hate Speech (2021).

In the paper, the authors train several models (including SVM, Bert, etc.) to complete two tasks: (1) distinguishing implicit hate speech from non-hate, and (2) categorizing implicit hate speech using one of the 6 classes in the fine-grained taxonomy and their best model is BERT based model. 

## Data Download Instruction:

1. Data used for the 6-way classification task: 
Per the authors' requirement in https://github.com/gt-salt/implicit-hate, you need to first complete a short survey through https://forms.gle/QxCpEbVp91Z35hWFA. Then follow this link to download: https://www.dropbox.com/s/24meryhqi1oo0xk/implicit-hate-corpus-nov-2021.zip?dl=0. (Specificall, we used the dataset "implicit_hate_v1_stg2_posts.tsv" in the zip file.) 

2. Data used for the binary classification task:
Since the author of the original paper does not provide Cancel changesthe dataset, we used another similar dataset accessible at https://www.kaggle.com/c/nlp-getting-started/data. (You also need to log in to Kaggle and accept the rules to download this dataset. Specificall, we used the dataset "train.csv" in the zip file.)

## File Description:

1. preprocessing.py: Python codes for cleaning and splitting the originl data set

2. Benchmark_Models.py: Python codes for implementing the benchmark models for classification including SVM(unigram), SVM(TF-IDF), SVM(GloVe), Naive Bayesian(TF-IDF), and Logistic Regression(TF-IDF) on the data set in the paper. (including training and evaluation) 

3. Bert_based_implicit_hate_classification.ipynb: Python (Google Colab) codes for implementing the BERT model for 6-way classification (including transformer, training and evaluation) 

4. dataset2.py: Python codes for implementing the benchmark models for classification including SVM(unigram), SVM(TF-IDF), SVM(GloVe) for binary classification on the extra dataset (including training and evaluation) 

5. Bert_based_disaster_classification.ipynb:  Python (Google Colab) codes for implementing the BERT model for binary classification on the extra dataset (including transformer, training and evaluation) 


## Table of Main Reproducibility Results:

1. Six-way Classification

| Models        | F1            | Accuracy      |
| ------------- | ------------- | ------------- |
| SVM(unigram)  | 52.6          | 53.0          |
| SVM(TF-IDF)   | 53.2          | 54.0          |
| SVM(GloVe)    | 38.5          | 39.8          |
| BERT          | 59.1          | 59.3          |

2. Binary Classification

| Models        | F1            | Accuracy      |
| ------------- | ------------- | ------------- |
| SVM(unigram)  | 78.7          | 78.9          |
| SVM(TF-IDF)   |79.3           | 79.7          |
| SVM(GloVe)    | 69.3          | 69.5          |
| BERT          | 81.0          | 80.1          |

