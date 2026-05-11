import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_redundant=0,
    n_informative=4,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_1 = DecisionTreeClassifier()

# model = SVC()
# model = KNeighborsClassifier()
model_2 = RandomForestClassifier()
# model = AdaBoostClassifier()
# model = LogisticRegression()
# model = GaussianNB()
# model = MLPClassifier(max_iter=1000)
# model = XGBClassifier(eval_metric='logloss')
# model = LGBMClassifier(verbose=-1)

model_1.fit(X_train, y_train)
pred_1 = model_1.predict(X_test)

model_2.fit(X_train, y_train)
pred_2 = model_2.predict(X_test)

print("Accuracy (Decision Tree) :", accuracy_score(y_test, pred_1))
print("Accuracy (Random Forest) :", accuracy_score(y_test, pred_2))
print("Confusion Matrix (Decision Tree) :")
print(confusion_matrix(y_test, pred_1))
print("Confusion Matrix (Random Forest) :")
print(confusion_matrix(y_test, pred_2))
print("Classification Report (Decision Tree) :")
print(classification_report(y_test, pred_1))
print("Classification Report (Random Forest) :")
print(classification_report(y_test, pred_2))

plt.figure(figsize=(6,5))
plt.bar(["Decision Tree", "Random Forest"], [accuracy_score(y_test, pred_1), accuracy_score(y_test, pred_2)])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")      
plt.show()

plt.scatter(X_test[:,0], X_test[:,1], c=pred_2, cmap='viridis')
plt.title("Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()