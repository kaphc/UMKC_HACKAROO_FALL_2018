import pandas as pd

df = pd.read_csv("data/Combined_News_DJIA_changed.csv")

date = []
label = []
news = []
category = []
for iter in range(0, df.__len__()):
    if df['category'][iter] == "Business & Finance":
        date.append(df['Date'][iter])
        label.append(df['Label'][iter])
        news.append(df['News'][iter])

data = {'Date': date, 'Label': label, 'News': news}
extracted_df = pd.DataFrame(data=data)
extracted_df.to_csv("data/Combined_News_DJIA_extracted.csv")