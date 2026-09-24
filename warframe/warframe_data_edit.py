from unicodedata import normalize
from utility import read_csv, save_csv


df = read_csv("data\\warframe\\warframe.csv", delimiter=';')


# Removes new lines from raw data
for x in df[1:]:
    # x[0] = int(x[0]) # Created
    # x[1] = int(x[1]) # Up
    x[2] = float(x[2].replace('%', '')) # Score
    # x[3] = int(x[3]) # Upvotes
    x[4] = x[4].replace('\n', '') # OriginalText
    x[5] = x[5].replace('\n', '') # EnglishTranslation

# Removes strange unicode characters from data
for x in df[1:]:
    # x[0] = int(x[0]) # Created
    # x[1] = int(x[1]) # Up
    # x[2] = float(x[2]) # Score
    # x[3] = int(x[3]) # Upvotes
    x[4] = normalize('NFKD', x[4]).encode('ascii','ignore').decode("utf-8") # OriginalText
    x[5] = normalize('NFKD', x[5]).encode('ascii','ignore').decode("utf-8") # EnglishTranslation

# Drop unneeded features from data and split class from features
df_features = []
df_class = []

for i, x in enumerate(df):
    # skip empty OriginalText rows
    if (x[4].strip() == ""):
        print(i)
        continue

    # x[0] = int(x[0]) # Created
    df_class.append(x[1]) # Up
    # x[2] = float(x[2]) # Score
    # x[3] = int(x[3]) # Upvotes
    df_features.append([x[4]]) # OriginalText
    # x[5] = x[5] # EnglishTranslation

save_csv("data\\warframe\\warframe_feature.csv", df_features)
df_class[0] = ["Up"]
save_csv("data\\warframe\\warframe_class.csv", df_class)
