from collections import Counter
import csv
import pickle

from sklearn.metrics import confusion_matrix


def clamp(value, min, max, err = 1e-6):
    if value > max - err:
        return max
    
    if value < min + err:
        return min
    
    return value

def class_rates(labels) -> tuple[str, str, int, int, float, float]:
    if type(labels) == list:
        labels = [l[0] for l in labels]

    counter = Counter(l for l in labels)
    (x1, x0), (y1, y0) = zip(*counter.most_common())
    
    return x1, x0, y1, y0, y1/(y1+y0)*100, y0/(y1+y0)*100

def read_csv(file_path, delimiter=','):
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return [x for x in reader]

def save_csv(file_path, data, delimiter=','):
    with open(file_path, "w") as f:
        w = csv.writer(f, delimiter=delimiter, lineterminator='\n')
        for x in data:
            w.writerow(x)

def strip_list(data):
    return [d[0] for d in data]

def save_model(file_path, model):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def cm(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted, labels=["1", "0"])
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
