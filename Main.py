#from matplotlib import pyplot
#from pandas import read_csv
#import pandas as pd

#path = r"Data.csv"
#headernames = ['overall','reviewText']

#data = read_csv(path, names=headernames)
#count = data.groupby('overall').size()
#print(count)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from pandas import DataFrame
from sklearn.feature_extraction import text
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from sklearn.model_selection import LeaveOneOut
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import spacy
from sklearn.metrics import plot_confusion_matrix

from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import gensim
all_stopwords = gensim.parsing.preprocessing.STOPWORDS


dataset = pd.read_csv('Final_data_22.csv')

counts = dataset['sentiment'].value_counts()
plt.bar(range(len(counts)), counts)
plt.show()
#print(counts)

#print(dataset)
#dataset.head()

#def preprocess_string(str_arg):



    #cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    #cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    #cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case

    #return cleaned_str # returning the preprocessed string

nlp = spacy.load(('en_core_web_lg') ,disable=['parser', 'ner'])

dataset['new_review'] = dataset['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#print(dataset['new_review'].head())
print(len(dataset['new_review']))






dataset['new_review'] = dataset['new_review'].apply(lambda x: " ".join(x for x in x.split() if x not in all_stopwords))
#print(dataset['new_review'].head())



dataset['new_review'] = dataset['new_review'].str.replace('[^\w\s]','',regex=True)
#print(dataset['new_review'].head())

#print(dataset['new_review'].head())
#dataset = [preprocess_string(review) for review in dataset]
#print(dataset['new_review'].head())
#print(dataset(2))
def space(comment):
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])
dataset['new_review']= dataset['new_review'].apply(space)
#####dataset.head()
#print(dataset['new_review'].head())

def tok(text):
    tokens = word_tokenize(text)
    return tokens

#count_vec = CountVectorizer( binary=True,lowercase=True, ngram_range=(1,1) , max_df =0.50 ,tokenizer=tok )
#count_vec.fit_transform(dataset['new_review'])
#count_vec_out = count_vec.transform(dataset.new_review)

#print(DataFrame(dataset))
#print(DataFrame(count_vec_out.A, columns=count_vec.get_feature_names()).to_string())


count_vec_standard = CountVectorizer( lowercase=True , ngram_range=(1,3), max_df =0.50 ,tokenizer=tok )
count_vec_standard.fit_transform(dataset['new_review'])
count_vec_out_standard = count_vec_standard.transform(dataset.new_review)
print(count_vec_out_standard.shape)

#print(DataFrame(count_vec_out_standard.A, columns=count_vec_standard.get_feature_names()).to_string())


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
#count_vec_standard_bigram.fit_transform(dataset['new_review'])
#count_vec_out_standard_bigram = count_vec_standard_bigram.transform(dataset.new_review)
#print(DataFrame(count_vec_out_standard.A, columns=count_vec_standard.get_feature_names()).to_string())


#X_train, X_test, y_train, y_test = train_test_split(count_vec_out_standard_Both, dataset['sentiment'], test_size=0.3, random_state=123)
#print(len(y_train))
#print(len(y_test))

#classifier_3  = MultinomialNB().fit(X_train, y_train)
#predicted_3= classifier_3.predict(X_test)
#print(y_test)
#print(predicted)
#print(classifier.score(X_test,y_test))
#print("%0.2f accuracy for only bigram" % (classifier_3.score(X_test,y_test)))

#tfidf = TfidfTransformer(use_idf=True)
#tfidf.fit(count_vec_out)
#tfidfed_out = tfidf.transform(count_vec_out)
#print(small_tfidfed)
#print(DataFrame(tfidfed_out.A, columns=count_vec.get_feature_names()).to_string())


#negative = dataset[dataset.sentiment == 2]
#neg_string = []
#for t in negative.new_review:
#    neg_string.append(t)
#neg_string = pd.Series(neg_string).str.cat(sep=' ')
#from wordcloud import WordCloud

#wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
#plt.figure(figsize=(12,10))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.show()





X_train, X_test, y_train, y_test = train_test_split(count_vec_out_standard, dataset['sentiment'], test_size=0.3, random_state=123)
print(len(y_train))
print(len(y_test))


#classifier  = MultinomialNB().fit(X_train, y_train)                                # standard
#predicted= classifier.predict(X_test)
#print(y_test)
#print(predicted)
#print(classifier.score(X_test,y_test))
#print("%0.2f accuracy for standard" % (classifier.score(X_test,y_test)))

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB



GNB = MultinomialNB()
classifier = GNB.fit(X_train, y_train)

predicted= GNB.predict(X_test)
accuracy_score = metrics.accuracy_score(classifier.predict(X_test),y_test)

print('GNB accuracy = ' + str('{:4.2f}'.format(accuracy_score*100))+'%')




#X_train, X_test, y_train, y_test = train_test_split(count_vec_out, dataset['sentiment'], test_size=0.3, random_state=123)
#print(len(y_train))
#print(len(y_test))

#classifier_1  = MultinomialNB().fit(X_train, y_train)
#predicted_1= classifier_1.predict(X_test)
#print(y_test)
#print(predicted)
#print(classifier_1.score(X_test,y_test))
#print("%0.2f accuracy for Binary" % (classifier_1.score(X_test,y_test)))
#print(classifier.score(X_train,y_train))

#print("----------------classification report----------------")


print(confusion_matrix(y_test, predicted))
#plot_confusion_matrix(classifier, X_test, y_test )
#plt.show()
target_names = ['Positive','Negative','Neurtal']
print(classification_report(y_test, predicted,target_names=target_names))


#######scores =cross_val_score(classifier, count_vec_out_standard, dataset['sentiment'], cv=10)
########print( scores )
############print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#scores_1 = cross_val_score(classifier, count_vec_out_standard,dataset['sentiment'], cv=10, scoring='f1_macro')
#print(scores_1)
#print("%0.2f F1_macro with a standard deviation of %0.2f" % (scores_1.mean(), scores_1.std()))

#cv = KFold(n_splits=10, random_state=123, shuffle=True)

#score_2 = cross_val_score(classifier, count_vec_out_standard, dataset['sentiment'], cv=cv)
#print(score_2)
#print("%0.2f accuracy for Kfold with a standard deviation of %0.2f" % (score_2.mean(), score_2.std()))


#cv_2=LeaveOneOut()
#score_3 =cross_val_score(classifier, count_vec_out_standard, dataset['sentiment'], cv=cv_2)
#print(score_3)

#print(mean(absolute(scores_3)))