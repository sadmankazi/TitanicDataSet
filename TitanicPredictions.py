import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree, model_selection

plt.close('all')

df = pd.read_csv("./data/train.csv")

# total number of females:
# num_females = df["Sex"].value_counts().values
# print(num_females[1])
#
# number of females survived:
# female_survived = df.Survived[df.Sex == "female"].value_counts()
# print(female_survived)

# print(df.shape)
# print(df.count())
# print(df.describe())

# NaN's in all columns
# count_nan = len(df) - df.count()
# print(count_nan)

# NaN's in Age column
# count_nan =len(df) - df['Age'].count()
# print(count_nan)

# Clean this data
df['Age'] = df['Age'].fillna((df['Age'].mean()))
df['Embarked'] = df['Embarked'].fillna('S')

df.loc[df["Sex"] == "male", "Sex"] = 0
df.loc[df["Sex"] == "female", "Sex"] = 1

df.loc[df["Embarked"] == "S", "Embarked"] = 0
df.loc[df["Embarked"] == "C", "Embarked"] = 1
df.loc[df["Embarked"] == "Q", "Embarked"] = 2

target = df['Survived'].values
featnames = ['Sex', 'Age', 'Pclass', 'Embarked', 'Fare']
features = df[featnames].values

decision_tree = tree.DecisionTreeClassifier(random_state=1, max_depth=7)
decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target))

scores = model_selection.cross_val_score(decision_tree, features, target, scoring='accuracy', cv=5)

print(scores)
print(scores.mean())

tree.export_graphviz(decision_tree_, feature_names=featnames, out_file="tree.dot")