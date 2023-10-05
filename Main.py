# Importing libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import gensim
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

# Loading and displaying dataset
dataset = pd.read_csv('Final_data_22.csv')
counts = dataset['sentiment'].value_counts()
plt.bar(range(len(counts)), counts)
plt.show()

# Text preprocessing
nlp = spacy.load(('en_core_web_lg'), disable=['parser', 'ner'])
all_stopwords = gensim.parsing.preprocessing.STOPWORDS

dataset['new_review'] = dataset['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dataset['new_review'] = dataset['new_review'].apply(lambda x: " ".join(x for x in x.split() if x not in all_stopwords))
dataset['new_review'] = dataset['new_review'].str.replace('[^\w\s]', '', regex=True)

def space(comment):
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])

dataset['new_review'] = dataset['new_review'].apply(space)

# Vectorization
def tok(text):
    tokens = word_tokenize(text)
    return tokens

count_vec_standard = CountVectorizer(lowercase=True, ngram_range=(1,3), max_df=0.50, tokenizer=tok)
count_vec_out_standard = count_vec_standard.fit_transform(dataset['new_review'])

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(count_vec_out_standard, dataset['sentiment'], test_size=0.3, random_state=123)

# Model training and evaluation
GNB = MultinomialNB()
classifier = GNB.fit(X_train, y_train)
predicted = GNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)
print('GNB accuracy = ' + str('{:4.2f}'.format(accuracy_score*100))+'%')

# Classification report
print("----------------classification report----------------")
print(confusion_matrix(y_test, predicted))
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

target_names = ['Positive', 'Negative', 'Neutral']
print(classification_report(y_test, predicted, target_names=target_names))

# Cross-validation
scores = cross_val_score(classifier, count_vec_out_standard, dataset['sentiment'], cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Original commented code
#from matplotlib import pyplot
#from pandas import read_csv
#import pandas as pd

#path = r"Data.csv"
#headernames = ['overall','reviewText']

#data = read_csv(path, names=headernames)
#count = data.groupby('overall').size()
#print(count)

#print(dataset)
#dataset.head()

#def preprocess_string(str_arg):
#    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
#    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
#    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
#    return cleaned_str # returning the preprocessed string

#print(dataset['new_review'].head())
#print(len(dataset['new_review']))

#print(dataset['new_review'].head())

#print(dataset['new_review'].head())
#dataset = [preprocess_string(review) for review in dataset]
#print(dataset['new_review'].head())
#print(dataset(2))

#count_vec = CountVectorizer( binary=True,lowercase=True, ngram_range=(1,1) , max_df =0.50 ,tokenizer=tok )
#count_vec.fit_transform(dataset['new_review'])
#count_vec_out = count_vec.transform(dataset.new_review)

#print(DataFrame(dataset))
#print(DataFrame(count_vec_out.A, columns=count_vec.get_feature_names()).to_string())

#count_vec_standard_Both = CountVectorizer( lowercase=True, stop_words='english', ngram_range=(1,2) , max_df =0.50 ,tokenizer=tok )
#count_vec_standard_Both.fit_transform(dataset['new_review'])
#count_vec_out_standard_Both = count_vec_standard_Both.transform(dataset.new_review)
#print(DataFrame(count_vec_out_standard.A, columns=count_vec_standard.get_feature_names()).to_string())

#X_train, X_test, y_train, y_test = train_test_split(count_vec_out_standard_Both, dataset['sentiment'], test_size=0.3, random_state=123)
#print(len(y_train))
#print(len(y_test))

#classifier_2  = MultinomialNB().fit(X_train, y_train)
#predicted= classifier_2.predict(X_test)
#print(y_test)
#print(predicted)
#print(classifier.score(X_test,y_test))
#print("%0.2f accuracy for unigrams and bigram" % (classifier_2.score(X_test,y_test)))

#count_vec_standard_bigram = CountVectorizer( lowercase=True, stop_words='english', ngram_range=(2,2) , max_df =0.50 ,tokenizer=tok )
#count_vec_standard
