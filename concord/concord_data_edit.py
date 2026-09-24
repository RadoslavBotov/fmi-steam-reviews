from unicodedata import normalize
from utility import read_csv, save_csv


df = read_csv("data\\concord\\concord.csv", delimiter=';')

# Removes new lines from raw data
for x in df[1:]:
    # x[0] = int(x[0]) # ReviewID
    # x[1] = x[1] # Language
    # x[2] = int(x[2]) # Up
    x[3] = float(x[3].replace('%', '')) # Score
    # x[4] = int(x[4]) # Upvotes
    x[5] = x[5].replace('\n', '') # OriginalText
    x[6] = x[6].replace('\n', '') # EnglishTranslation

# Removes strange unicode characters from data
for x in df[1:]:
    # x[0] = int(x[0]) # ReviewID
    # x[1] = x[1] # Language
    # x[2] = int(x[2]) # Up
    # x[3] = float(x[3]) # Score
    # x[4] = int(x[4]) # Upvotes
    x[5] = normalize('NFKD', x[5]).encode('ascii','ignore').decode("utf-8") # OriginalText
    x[6] = normalize('NFKD', x[6]).encode('ascii','ignore').decode("utf-8") # EnglishTranslation

# Drop unneeded features from data and split class from features
df_features = []
df_class = []

for x in df:
    # x[0] = int(x[0]) # ReviewID
    # x[1] = x[1] # Language
    df_class.append(x[2]) # Up
    # x[3] = float(x[3]) # Score
    # x[4] = int(x[4]) # Upvotes
    df_features.append([x[5]]) # OriginalText
    # x[6] = x[6] # EnglishTranslation

save_csv("data\\concord\\concord_feature.csv", df_features)
df_class[0] = ["Up"] # up -> Up
save_csv("data\\concord\\concord_class.csv", df_class)
