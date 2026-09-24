import numpy as np
import tqdm
from utility import class_rates, read_csv, save_model, strip_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def train_test(I_train, I_test, X):
    X_train = [X[i] for i in I_train]
    X_test = [X[i] for i in I_test]
    return X_train, X_test

# load data
df_feature = read_csv("data\\concord\\concord_feature.csv")
df_class = read_csv("data\\concord\\concord_class.csv")
df_emotions = read_csv("data\\concord\\concord_emotions.csv")

# remove feature name
df_feature = df_feature[1:]
df_class = df_class[1:]
df_emotions = df_emotions[1:]

# transform from list[list[str]] -> list[str]
df_class = strip_list(df_class)
df_feature = strip_list(df_feature)

# vectorize text data
vectorizer = CountVectorizer(stop_words='english')
# vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_feature).toarray()
y = [1 if l == "1" else 0 for l in df_class]

# Convert emotions to ints and concatenate to vectorized features
df_emotions = [
    [np.ceil(float(x)) for x in review_emotions]
    for review_emotions
    in df_emotions
]

X_emo = np.concatenate((X, df_emotions), axis=1)

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
averages_emo = []
best_model = None
best_model_emo = None
best_acc = 0
best_acc_emo = 0
rs = 42
for _ in tqdm.tqdm(range(100)):
    indexes = [i for i in range(len(df_class))]
    I_train, I_test, y_train, y_test = train_test_split(indexes, y, test_size=0.3, random_state=rs)
    rs += 1
    X_train, X_test = train_test(I_train, I_test, X)
    X_emo_train, X_emo_test = train_test(I_train, I_test, X_emo)

    # print(class_rates(y_train, None, 1, 0))
    # print(class_rates(y_test, None, 1, 0))

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    averages.append(acc)

    mnb_emo = MultinomialNB()
    mnb.fit(X_emo_train, y_train)
    predictions_emo = mnb.predict(X_emo_test)
    acc_emo = accuracy_score(y_test, predictions_emo)
    averages_emo.append(acc_emo)

    if (acc > best_acc):
        best_acc = acc
        best_model = mnb

    if (acc_emo > best_acc_emo):
        best_acc_emo = acc_emo
        best_model_emo = mnb_emo
    # print("  ", ave)

print(f"Average accuracy: {np.average(averages):2f}%")
print(f"Best accuracy:    {best_acc:2f}%")
print(f"Average emo accuracy: {np.average(averages_emo):2f}%")
print(f"Best emo accuracy:    {best_acc_emo:2f}%")
# save_model("results\\concord_best_nb_model_100.pkl", best_model)
