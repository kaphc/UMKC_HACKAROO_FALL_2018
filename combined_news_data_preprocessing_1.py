import pandas
from sklearn.externals import joblib

vectorizer = joblib.load('models/vectorizer.joblib')
news_text_classification_model = joblib.load('models/news_text_classification_model.joblib')
encoder = joblib.load('models/encoder.joblib')

date = []
label = []
news = []
combined_news_data = pandas.read_csv("data/Combined_News_DJIA.csv")
combined_news_data = combined_news_data.fillna("")

for i in range(0, combined_news_data.__len__()):
    for top_news in range(1, 25):
        if not combined_news_data["Top" + str(top_news)][i]:
            news.append("Unavailable")
        else:
            news.append(combined_news_data["Top" + str(top_news)][i][2:-1].lower())
        date.append(combined_news_data['Date'][i])
        label.append(combined_news_data['Label'][i])

data = {'Date': date, 'Label': label, 'News': news}
df = pandas.DataFrame(data=data)

news = df['News'].tolist()
news_vector = vectorizer.transform(iter(news))
category_prediction = news_text_classification_model.predict(news_vector)
df['category'] = encoder.classes_[category_prediction]

df.to_csv("data/Combined_News_DJIA_changed.csv")
