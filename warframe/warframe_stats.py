from collections import Counter

import numpy as np
import tqdm

from utility import read_csv, strip_list
from sklearn.feature_extraction.text import CountVectorizer

# class distribution
df_class = read_csv("data\\warframe\\warframe_class.csv")
df_class = df_class[1:]
df_class = strip_list(df_class)

counter = Counter(df_class)
(x1, x2), (y1, y2) = zip(*counter.most_common())

print(f"Upvote:   {y1} {y1/(y1+y2)*100:.2f}%")
print(f"Downvote: {y2} {y2/(y1+y2)*100:.2f}%")
print(f"          {y1+y2}")

# average length of reviews
df_feature = read_csv("data\\warframe\\warframe_feature.csv")
df_feature = df_feature[1:]
df_feature = strip_list(df_feature)

average = []
for x in df_feature:
    length = len(x)
    average.append(length)
print("Average length (symbol): ", np.average(average))

# vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_feature).toarray()

average = []
for x in tqdm.tqdm(X):
    words = 0
    for w in x:
        words += w
    average.append(words)
print("Average length (word): ", np.average(average))
