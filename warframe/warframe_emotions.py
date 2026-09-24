from LeXmo import LeXmo
import tqdm
from utility import read_csv, save_csv, strip_list


# load data
df_feature = read_csv("data\\warframe\\warframe_feature.csv")

# remove feature name
df_feature = df_feature[1:]

# transform from list[list[str]] -> list[str]
df_feature = strip_list(df_feature)

# feature names
df_emotions = [
    [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
        "positive",
        "negative"
    ]
]

# get feature values
for x in tqdm.tqdm(df_feature):
    emo = LeXmo.LeXmo(x)

    emotions = []
    emotions.append(emo.get("anger"))
    emotions.append(emo.get("anticipation"))
    emotions.append(emo.get("disgust"))
    emotions.append(emo.get("fear"))
    emotions.append(emo.get("joy"))
    emotions.append(emo.get("sadness"))
    emotions.append(emo.get("surprise"))
    emotions.append(emo.get("trust"))
    emotions.append(emo.get("positive"))
    emotions.append(emo.get("negative"))

    df_emotions.append(emotions)

save_csv("data\\warframe\\warframe_emotions.csv", df_emotions)
