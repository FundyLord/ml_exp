import numpy as np
import pandas as pd
from sklearn.datasets import  make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_classes=2,
    n_redundant=0,
    random_state=42
)

import numpy as np

X = np.transpose(X)
df = pd.DataFrame(
    {
        "f1": X[0],
        "f2": X[1],
        "label": y
    }
)

print(df.head())
print(df.describe())
print(df.info())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train,X_test,y_train,y_test = train_test_split(df[["f1" ,"f2"]],df["label"],test_size=0.2,random_state=42)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(
    n_neighbors=5,
)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.inspection import DecisionBoundaryDisplay
DecisionBoundaryDisplay.from_estimator(
    knn , X_train , response_method="predict" , alpha=0.5
)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
plt.title("KNN Decision Boundary")
plt.show()