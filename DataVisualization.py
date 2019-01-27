import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.close('all')

df = pd.read_csv("./data/train.csv")

# print(df.shape)
# print(df.count())
# print(df.describe())

fig = plt.figure(figsize=(12, 6))
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.5

# Survived vs Deceased
plt.subplot2grid((3, 4), (0, 0))
df.Survived.value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Total Survival")
plt.xticks(np.arange(2), ('Dead', 'Alive'),rotation=0)
plt.ylabel('%')

# Male survived vs deceased
plt.subplot2grid((3, 4), (0, 1))
df.Survived[df.Sex == "male"].value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Male Survival")
plt.xticks(np.arange(2), ('Dead', 'Alive'), rotation=0)
plt.ylabel('%')

# Female survived vs  deceased
plt.subplot2grid((3, 4), (0, 2))
df.Survived[df.Sex == "female"].value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Female Survival")
plt.xticks(np.arange(2), ('Alive', 'Dead'), rotation=0)
plt.ylabel('%')


# gender distribution of survival
plt.subplot2grid((3, 4), (0, 3))
df[df.Survived == 1].Sex.value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Gender distribution of Survival")
plt.xticks(np.arange(2), ('Female', 'Male'), rotation=0)
plt.ylabel('%')

# survival distribution within ticket class
plt.subplot2grid((3, 4), (1, 0), colspan=4)
for x in [1, 2, 3]:
    df.Survived[df.Pclass == x].plot(kind="kde")
plt.title("Survival Distribution within Classes")
plt.legend(("1st Class", "2nd Class", "3rd Class"), frameon=False)

# Low class male survived vs deceased
plt.subplot2grid((3, 4), (2, 0))
df.Survived[(df.Sex == "male") & (df.Pclass == 3)].value_counts(normalize='True').plot(kind='bar', color="lightblue", alpha=alpha_bar_chart)
plt.title("Low class Male Survival")
plt.xticks(np.arange(2), ('Dead', 'Alive'), rotation=0)
plt.ylabel('%')


# high class male survived vs deceased
plt.subplot2grid((3, 4), (2, 1))
df.Survived[(df.Sex == "male") & (df.Pclass == 1)].value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("High class Male Survival")
plt.xticks(np.arange(2), ('Dead', 'Alive'), rotation=0)
plt.ylabel('%')

# low class female survived vs deceased
plt.subplot2grid((3, 4), (2, 2))
df.Survived[(df.Sex == "female") & (df.Pclass == 3)].value_counts(normalize='True').plot(kind='bar', color="pink", alpha=alpha_bar_chart)
plt.title("Low class Female Survival")
plt.xticks(np.arange(2), ('Alive', 'Dead'), rotation=0)
plt.ylabel('%')


# high class female survived vs  deceased
plt.subplot2grid((3, 4), (2, 3))
df.Survived[(df.Sex == "female") & (df.Pclass == 1)].value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("High class Female Survival")
plt.xticks(np.arange(2), ('Alive', 'Dead'), rotation=0)
plt.ylabel('%')


plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
plt.show()


fig = plt.figure(figsize=(12, 6))

# Survived vs Deceased
plt.subplot2grid((2, 3), (0, 0))
df.Survived.value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Total Survival")
plt.xticks(np.arange(2), ('Dead', 'Alive'),rotation=0)
plt.ylabel('%')

# Male survived vs deceased
plt.subplot2grid((2, 3), (0, 1))
plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Survival by Age")
plt.xticks(np.arange(2), ('Dead', 'Alive'),rotation=0)
plt.ylabel('%')

# Male survived vs deceased
plt.subplot2grid((2, 3), (0, 2))
df.Pclass.value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Survival by Class")
plt.ylabel('%')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
for x in [1, 2, 3]:
    df.Age[df.Pclass == x].plot(kind='kde')
plt.title("Class wrt Age")
plt.legend(("1st Class", "2nd Class", "3rd Class"), frameon=False)


plt.subplot2grid((2, 3), (1, 2))
df.Embarked.value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Survival by Class")
plt.ylabel('%')

plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
plt.show()