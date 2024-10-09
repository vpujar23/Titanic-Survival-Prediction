# Simranjit Bhella & Shishir Gururaj 2023
#
# Titanic survival prediction using logistic regression

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the preprocessed train and test datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Separate features and target in the training data
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]

# Separate features and target in the test data
X_test = test_data.drop("Survived", axis=1)
y_test = test_data["Survived"]

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)


def evaluate_model(predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    return accuracy, precision, recall, f1


logreg_accuracy, logreg_precision, logreg_recall, logreg_f1 = evaluate_model(
    logreg_predictions, y_test)

# Print results
print("Logistic Regression Results:")
print(f"Accuracy: {logreg_accuracy:.2f}")
print(f"Precision: {logreg_precision:.2f}")
print(f"Recall: {logreg_recall:.2f}")
print(f"F1-score: {logreg_f1:.2f}")

# Visualization: Gender vs Survived (Number of Passengers)
gender_survival = train_data.groupby(['Sex', 'Survived']).size().unstack()
gender_survival.rename(
    {0: 'Not Survived', 1: 'Survived'}, axis=1, inplace=True)
ax = gender_survival.plot(kind='bar', stacked=True,
                          color=['lightgray', '#0072BD'])
ax.set_xticklabels(['Male', 'Female'], rotation=0)
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.title('Survived vs Sex')
plt.legend(title='Survival Status')
plt.show()
