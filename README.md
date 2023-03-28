Sentiment Analysis using Naive Bayes Classifier:
This Python script demonstrates sentiment analysis using the Naive Bayes classifier on a dataset of text reviews. The script preprocesses the text data, tokenizes and vectorizes it, trains a Naive Bayes classifier, and evaluates its performance.
Libraries used : 
The script uses the following libraries:
* pandas
* numpy
* sklearn
* re
* nltk
* matplotlib
* wordcloud
* spacy
* gensim

Data :
The dataset is expected to be in a CSV file called 'Final_data_22.csv', with a column named 'review' for the text reviews and a column named 'sentiment' for the sentiment labels.

Preprocessing:
The text data undergoes the following preprocessing steps:
1. Convert all text to lowercase.
2. Remove stopwords using the Gensim library.
3. Remove special characters and punctuation.
4. Lemmatize words using the Spacy library.

Feature extraction:
The preprocessed text is tokenized and transformed into a document-term matrix using the CountVectorizer function from the sklearn.feature_extraction.text module. The resulting matrix is then used as input for training the Naive Bayes classifier.

Model training and evaluation:
The script trains a Naive Bayes classifier using the MultinomialNB function from the sklearn.naive_bayes module. The dataset is split into training and testing sets using the train_test_split function from the sklearn.model_selection module.
The classifier's performance is evaluated using various metrics, including accuracy, precision, recall, and F1-score. A confusion matrix is also generated for additional insights into the model's performance.

Visualization:
A bar plot is generated using the Matplotlib library to visualize the distribution of sentiment labels in the dataset.
