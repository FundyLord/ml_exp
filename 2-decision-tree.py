import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X,y = make_classification(
    n_samples=200,
    n_features=4,
    n_classes=2,
    random_state=42
)

X = np.transpose(X)

dictionary = {
    "f1": X[0],
    "f2": X[1],
    "f3": X[2],
    "f4": X[3],
    "target": y
}

df = pd.DataFrame(dictionary)
print(df.head())
print(df.describe())
print(df.info())

X_train, X_test, y_train,y_test = train_test_split(df[["f1","f2","f3","f4"]], df["target"], test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    # criterion = "gini",
    max_depth=3,
)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plot_tree(
    decision_tree ,
          feature_names=["f1","f2","f3","f4"] ,
          class_names=["0" , "1"] ,
          filled=True
          )
plt.show()
