import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("./data/train.csv")

# print(df.shape)
# print(df.count())
# print(df.describe())

fig = plt.figure(figsize=(18, 6))
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.5

# Survived vs Deceased
fig.add_subplot(3, 4, 1)
df.Survived.value_counts(normalize='True').plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Survival")
plt.xticks(np.arange(2), ('Dead', 'Alive'),rotation=0)
plt.ylabel('%')
plt.show()


