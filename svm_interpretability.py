import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
import matplotlib.pyplot as plt

stopwords_file = '../data/sl_stopwords.txt' # file with stopwords lists

with open(stopwords_file, encoding='utf8') as file: # reading the stopwords file and creating a list
    lines = file.readlines()
    stopwords = [line.rstrip() for line in lines]

df = pd.read_csv('../data/speeches.csv') # reading dataset with speeches of politicians and their political orientation

x = df['Speech text']
y = df['Political orientation'] 

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.35, min_df=5, stop_words=stopwords) # initialize tfidf vectorizer
x_tfidf = vectorizer.fit_transform(x) # fit speeches and transform

svc = svm.SVC(kernel='linear', C=1) # initialize Linear SVM Classifier
svc.fit(x_tfidf, y) # fit the model

features_names = vectorizer.get_feature_names() # get feature names (tf-idf uni, bi and tri-grams) for plotting purposes

coefs = svc.coef_.toarray()[0]

pd.Series(coefs, index=features_names).nlargest(20).plot(kind='barh') # plot the tf-idf features with highest importance (class=1, right-wing politicians)

pd.Series(coefs, index=features_names).nsmallest(20).plot(kind='barh') # plot the tf-idf features with highest importance (class=-1, left-wing politicians)