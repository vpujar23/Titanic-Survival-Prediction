import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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

# Decision Tree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
dt_predictions = decision_tree_model.predict(X_test)

# Model evaluation


def evaluate_model(predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    return accuracy, precision, recall, f1


logreg_accuracy, logreg_precision, logreg_recall, logreg_f1 = evaluate_model(
    logreg_predictions, y_test)
dt_accuracy, dt_precision, dt_recall, dt_f1 = evaluate_model(
    dt_predictions, y_test)

# ROC Curve for Decision Tree
fpr, tpr, _ = roc_curve(
    y_test, decision_tree_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Print results
print("Logistic Regression Results:")
print(f"Accuracy: {logreg_accuracy:.2f}")
print(f"Precision: {logreg_precision:.2f}")
print(f"Recall: {logreg_recall:.2f}")
print(f"F1-score: {logreg_f1:.2f}")

print("\nDecision Tree Results:")
print(f"Accuracy: {dt_accuracy:.2f}")
print(f"Precision: {dt_precision:.2f}")
print(f"Recall: {dt_recall:.2f}")
print(f"F1-score: {dt_f1:.2f}")
