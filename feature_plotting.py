import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

detail = {"age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure",
          "chol": "Serum Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
          "thalach": "Max Heart Rate", "exang": "Exercise Induced Angina", "oldpeak": "Oldpeak",
          "slope": "Slope", "ca": "Number of major vessels", "thal": "Thal", "target": "(0 - no disease, 1 - disease))"}

# Age vs. All Real for both Sexes
# for feature in data.columns:
#     if feature in realFeatures:
#         plt.figure()
#         sns.relplot(    
#                 data=data, x="age", y=feature, col="sex",
#                 hue="target"
#         )
#         plt.savefig(f"plots/{feature}_vs_age.png")

# Categorical Counts
# for feature in data.columns:
#     if feature in categoricalFeatures:
#         plt.figure()
#         sns.countplot(  # histplot if for continuous non categorical data
#             data=data, x=feature, hue="target"
#         )
#         plt.title(f"{detail[feature]} {detail['target']}")
#         plt.savefig(f"plots/{feature}_count.png")

# Categorical Normalized Counts
# for feature in data.columns:
#     if feature in categoricalFeatures:
#         # thanks copilot
        # proportions = data.groupby(feature)["target"].value_counts(normalize=True).rename("proportion").reset_index()

#         plt.figure()
#         sns.barplot(data=proportions, x=feature, y="proportion", hue="target")
#         plt.title(f"{detail[feature]} {detail['target']}")
#         plt.show()

no_col = 10
cmaps = ("crest", "YlOrBr", "Oranges", "rocket", "flare")
mosaic = pd.DataFrame(data=[np.random.random(no_col) for x in range(100)], columns=[x for x in range(no_col)])



# print(data_default.head())
plt.figure(figsize=(13, 10))

for i in range(5):
    plt.figure()
    sns.heatmap(mosaic.corr(), annot=False, linewidths=2, cmap=cmaps[i])
    plt.savefig(f"plots/oneHot_heatmap{i}.png")
plt.show()