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
from lightgbm import LGBMClassifier

X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model = DecisionTreeClassifier()

# model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()
# model = AdaBoostClassifier()
# model = LogisticRegression()
# model = GaussianNB()
# model = MLPClassifier(max_iter=1000)
# model = XGBClassifier(eval_metric='logloss')
# model = LGBMClassifier(verbose=-1)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print("Confusion Matrix :")
print(confusion_matrix(y_test, pred))
print("Classification Report :")
print(classification_report(y_test, pred))

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.title("Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()