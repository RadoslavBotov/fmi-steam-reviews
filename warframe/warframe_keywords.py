from utility import class_rates, read_csv, strip_list
from nltk.tokenize import word_tokenize
from rake_nltk import Rake

# load data
df_class = read_csv("data\\warframe\\warframe_class.csv")
df_feature = read_csv("data\\warframe\\warframe_feature.csv")

# remove feature name
df_class = df_class[1:]
df_feature = df_feature[1:]

# remove list
df_class = strip_list(df_class)
df_feature = strip_list(df_feature)

# get original dataset class distribution
c1, c0, y1, y0, p1, p0 = class_rates([x[0] for x in df_class])
print(c1, c0, y1, y0, p1, p0)

# 
df_feature = [word_tokenize(review) for review in df_feature]
df_feature = [[word.lower() for word in review] for review in df_feature]
df_feature = [" ".join(review) for review in df_feature]

# split positive and negative reviews
df_feature_p = [f for f, l in zip(df_feature, df_class) if l[0] == "1"]
df_feature_n = [f for f, l in zip(df_feature, df_class) if l[0] == "0"]

r = Rake()

r.extract_keywords_from_sentences(df_feature_p)
keywords_p = r.get_ranked_phrases_with_scores()

r.extract_keywords_from_sentences(df_feature_n)
keywords_n = r.get_ranked_phrases_with_scores()

for (s1, p1), (s2, p2) in zip(keywords_p[:50], keywords_n[:50]):
    print(f"{s1:<4.2f}|{p1:<60}|{s2:<4.2f}|{p2:<60}")
