import numpy as np
import tqdm
from utility import class_rates, read_csv, save_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load data
df_feature = read_csv("data\\warframe\\warframe_feature.csv")
df_class = read_csv("data\\warframe\\warframe_class.csv")

# remove feature name
df_feature = df_feature[1:]
df_class = df_class[1:]

# vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X_feature = [x[0] for x in df_feature]
X = vectorizer.fit_transform(X_feature).toarray()
y = [1 if x[0] == "1" else 0 for x in df_class]

print(len(X[0]))

# X_train -> Training split with features
# X_test  -> Testing split with features
# y_train -> Training with class labels
# y_test  -> Testing with class labels

# print(len(X_train), len(X_test))
# print(len(y_train), len(y_test))

# get label distribution in original dataset
print(class_rates(df_class))

# make N experiments for random subsets of original dataset for Multinomial Naive Bayes Classifier
averages = []
best_model = None
best_acc = 0
for _ in tqdm.tqdm(range(100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # print(class_rates(y_train, None, 1, 0))
    # print(class_rates(y_test, None, 1, 0))

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    averages.append(acc)

    if (acc > best_acc):
        best_acc = acc
        best_model = mnb
    # print("  ", ave)

print(f"Average accuracy: {np.average(averages)*100:2f}%")
print(f"Best accuracy:    {best_acc*100:2f}%")

# save_model("nb_models\\warframe_best_nb_model_100.pkl", best_model)
