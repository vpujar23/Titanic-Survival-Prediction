import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data reading
df = pd.read_csv("train.csv")

# Data preprocessing
mean = df['Age'].mean()

df['Age'].fillna(mean, inplace=True)

Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass',
                        drop_first=True)  # Create dummies for Pclass
Sex = pd.get_dummies(df['Sex'], drop_first=True)  # Create dummies for Sex

# Combining the preprocessed features
df = pd.concat([df, Pclass, Sex], axis=1)
df.drop(['PassengerId', 'Sex', 'Age', 'Pclass', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        axis=1, inplace=True)

# Converting column names to strings
df.columns = df.columns.astype(str)

x = df.drop('Survived', axis=1)
y = df['Survived']

# Model implementation
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=6)
reg = LogisticRegression(max_iter=1000)  # Adding maximum iterations
reg.fit(x_train, y_train)
pred = reg.predict(x_test)

# Accuracy
print("Logistic Regression Results:")
accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy*100:.2f}%")
