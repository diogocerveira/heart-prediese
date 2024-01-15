import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Load the data from the CSV file
data = pd.read_csv('H:\\O meu disco\\4º Ano\\1º Semestre\\DACO\\Project\\heart-prediese\\heart.csv')
# Load the data from the DAT file
#data = pd.read_csv('heart.dat', sep='\\s+')


detail = {"age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure",
          "chol": "Serum Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
          "thalach": "Max Heart Rate", "exang": "Exercise Induced Angina", "oldpeak": "Oldpeak",
          "slope": "Slope", "ca": "CA", "thal": "Thal", "target": "(0 - no disease, 1 - disease))"}

# Separate the features (first 13 columns) and the target variable (last column)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets using cross-validation


# Create empty lists to store the evaluation metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

print(X)
print()
print(y)


for random_state in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # Perform further analysis or model training with the current split
    # Create a Naive Bayes classifier
    model = GaussianNB()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    print(y_pred)

    # Calculate evaluation metrics
    # pos_label referes to the HD presence class (the positive class)
    pos_label = 1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=pos_label)
    recall = recall_score(y_test, y_pred, pos_label=pos_label)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Append the evaluation metrics to the respective lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Print the evaluation metrics for the current iteration
    print("Iteration:", random_state + 1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, columns=["Predicted HD Absence", "Predicted HD Presence"], index=["Actual HD Absence", "Actual HD Presence"]))
    print()


# Create a DataFrame to store the evaluation metrics for each iteration
metrics_df = pd.DataFrame({
    "Iteration": range(1, 6),
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1 Score": f1_list
})

# Print the metrics table
print("Metrics Table:")
print(metrics_df)
