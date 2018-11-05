import nltk
import pandas
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("\nLoading data.....")
data = pandas.read_csv('data/Combined_News_DJIA_changed.csv')
X = data['News']
y = data['Label']

print("\nSplitting data.....")
news_train, news_test, label_train, label_test = train_test_split(X, y)
print("training set size:" + str(len(news_train)))
print("testing set size:" + str(len(news_test)))

print("\nVectorizing data.....")
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)

print("\nVectorizing predictors data.....")
vectorizer.fit(iter(news_train))
X_train = vectorizer.transform(iter(news_train))
X_test = vectorizer.transform(iter(news_test))

print("\nTraining Multinomial Naive Bayesian......")
news_text_classification_model = MultinomialNB()
news_text_classification_model.fit(X_train, label_train)
Y_pred = news_text_classification_model.predict(X_test)
print(accuracy_score(label_test, Y_pred))
print(classification_report(label_test, Y_pred))
