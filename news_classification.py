import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib

categories = ['Business & Finance', 'Health Care', 'Science & Health', 'Politics & Policy', 'Criminal Justice']

print("\nLoading data.....")
titles = []
categories_from_news_data = []
with open('data/dsjVoxArticles.tsv','r', encoding='utf8') as tsv:
    for line in tsv:
        a = line.strip().split('\t')[:3]
        if a[2] in categories:
            title = a[0].lower()
            title = re.sub('\s\W',' ',title)
            title = re.sub('\W\s',' ',title)
            titles.append(title)
            categories_from_news_data.append(a[2])

print("\nSplitting data.....")
title_train, title_test, category_train, category_test = train_test_split(titles,categories_from_news_data)
print("training set size:" + str(len(title_train)))
print("testing set size:" + str(len(title_test)))

print("\nVectorizing data.....")
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)

print("\nVectorizing predictors data.....")
vectorizer.fit(iter(title_train))
joblib.dump(vectorizer, 'models/vectorizer.joblib')
X_train = vectorizer.transform(iter(title_train))
X_test = vectorizer.transform(iter(title_test))

print("\nEncoding response data.....")
encoder = LabelEncoder()
encoder.fit(category_train)
print(encoder.classes_)
joblib.dump(encoder, 'models/encoder.joblib')
Y_train = encoder.transform(category_train)
Y_test = encoder.transform(category_test)

print("\nTraining Multinomial Naive Bayesian......")
news_text_classification_model = MultinomialNB()
news_text_classification_model.fit(X_train, Y_train)
Y_pred = news_text_classification_model.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred, target_names=encoder.classes_))

print("\nTo store the trained model......")
joblib.dump(news_text_classification_model, 'models/news_text_classification_model.joblib')