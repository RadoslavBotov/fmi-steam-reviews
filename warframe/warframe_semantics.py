from sklearn.metrics import accuracy_score, confusion_matrix
import tqdm
from utility import class_rates, read_csv, strip_list

def predict(x, threshold=0):
    if (x[8] - x[9] >= threshold - 1e-6):
        return "1"
    
    return "0"

# load data
df_class = read_csv("data\\warframe\\warframe_class.csv")
df_emotions = read_csv("data\\warframe\\warframe_emotions.csv")

# remove feature name
df_class = df_class[1:]
df_emotions = df_emotions[1:]

# transform from list[list[str]] -> list[str]
df_class = strip_list(df_class)
df_emotions = [[float(x) for x in review] for review in df_emotions]

# get original dataset class distribution
c1, c0, y1, y0, p1, p0 = class_rates(df_class)
print(c1, c0, y1, y0, p1, p0)

# split positive and negative reviews
df_emotion_p = [f for f, l in zip(df_emotions, df_class) if l == "1"]
df_emotion_n = [f for f, l in zip(df_emotions, df_class) if l == "0"]

# experiments over ALL reviews
predictions_all = []
best_threshold = 0
best_acc = 0
for k in tqdm.tqdm(range(0, 100, 1)):
    for x in df_emotions:
        predictions_all.append(predict(x, - k / 100))
    acc = accuracy_score(df_class, predictions_all)
    if (acc > best_acc):
        best_acc = acc
        best_threshold = - k / 100
    predictions_all = []

print(best_threshold * 100, best_threshold)
print(best_acc)

threshold = 0
threshold = best_threshold

# experiments over ALL reviews
predictions_all = []
for x in df_emotions:
    predictions_all.append(predict(x, threshold))
cm = confusion_matrix(df_class, predictions_all, labels=["1", "0"])
# TP, FN
# FP, TN
TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]
print(cm)
print("Accuracy:    ", (TP + TN) / (TP + FN + FP + TN))
print("Precision:   ", TP / (TP + FP))
print("Recall:      ", TP / (TP + FN))
print("Specificity: ", TN / (TN + FP))
