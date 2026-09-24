from utility import cm, read_csv, class_rates, strip_list

# load data
df_class = read_csv("data\\concord\\concord_class.csv")
df_feature = read_csv("data\\concord\\concord_feature.csv")

# remove feature name
df_class = df_class[1:]
df_feature = df_feature[1:]

# remove feature name
df_class = strip_list(df_class)
df_feature = strip_list(df_feature)

# get original dataset class distribution
l1, l0, y1, y0, c1, c0 = class_rates(df_class)

if (y1 > y0):
    goal = l1
else:
    goal = l0

#
predictions = []
for x in df_feature:
    predictions.append(goal)

cm(df_class, predictions)
