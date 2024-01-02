import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

'''
In the context of data types, "ordered" data refers to ordinal data.
Ordinal data is a type of categorical data with an order (or rank).
The order of these values is significant and typically represents some sort of hierarchy.
For example, ratings data (like "poor", "average", "good", "excellent") is ordinal
because there is a clear order to the categories.
'''
'''
1. age - Real
2. sex - Binary
3. cp - Chest pain type (4 values) - Nominal
4. trestbps - Resting blood age - Real
5. chol - Serum cholesterol (in mg/dl) - Real
6. fbs - Fasting blood sugar > 120 mg/dl - Binary
7. restecg - Resting electrocardiographic results (values 0,1,2) - Nominal
8. thalach - Maximum heart rate achieved - Real
9. exang - Exercise induced angina - Binary
10. oldpeak - Oldpeak = ST depression induced by exercise relative to rest - Real
11. slope - The slope of the peak exercise ST segment - Ordered
12. ca - Number of major vessels (0-3) colored by flouroscopy - Real
13. thal - Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect - Nominal
14. target: 1 = no disease; 2 = presence of disease
'''

detail = {"age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure",
          "chol": "Serum Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
          "thalach": "Max Heart Rate", "exang": "Exercise Induced Angina", "oldpeak": "Oldpeak",
          "slope": "Slope", "ca": "CA", "thal": "Thal", "target": "(0 - no disease, 1 - disease))"}



sns.set_theme(context="paper", font_scale=1.5, style="whitegrid", palette="Set2")

data = pd.read_csv("heart.csv")
# drop non-real attributes
# data.drop(["cp", "fbs", "restecg", "exang", "slope", "thal"], axis=1, inplace=True)
noFeatures = data.shape[1]
print("Number of features: ", noFeatures)

realFeatures = ("trestbps", "chol", "thalach", "oldpeak")
# considering ca ordered as only has 4 values
categoricalFeatures = ("cp", "fbs", "restecg", "exang", "slope", "thal", "ca")
extraFeatures = ("age", "sex", "target")

# Age vs. All Real for both Sexes
for feature in data.columns:
    if feature in realFeatures:
        plt.figure()
        sns.relplot(    
                data=data, x="age", y=feature, col="sex",
                hue="target"
        )
        plt.savefig(f"plots/{feature}_vs_age.png")
    

# Categorical Counts
for feature in data.columns:
    if feature in categoricalFeatures:
        plt.figure()
        sns.countplot(  # histplot if for continuous non categorical data
            data=data, x=feature, hue="target"
        )
        plt.title(f"{detail[feature]} {detail['target']}")
        plt.savefig(f"plots/{feature}_count.png")

# Categorical Normalized Counts
# for feature in data.columns:
#     if feature in categoricalFeatures:
#         # thanks copilot
#         proportions = data.groupby(feature)["target"].value_counts(normalize=True).rename("proportion").reset_index()

#         plt.figure()
#         sns.barplot(data=proportions, x=feature, y="proportion", hue="target")
#         plt.title(f"{detail[feature]} {detail['target']}")
#         plt.show()

# for feature in data.columns[:-1]:
#     plotter2v2(data, feature, "target")

# normalizer = StandardScaler()
# calculate and remove mean and standard deviation from the data
# data_scaled = pd.DataFrame(normalizer.fit_transform(data), columns=data.columns)



# print(data.head())
# sns.heatmap(data_scaled.corr(), annot=True, linewidths=2)
# plt.show()
