import numpy as np

"""""
Возьмите датасет Mnist (рукописные цифры от 0 до 9) и используйте каждый из
известных вам классификаторов, сравните качество классификации, объясните
почему одни из классификаторов работают лучше или хуже.

KNN, decision tree, SVM, log reg
"""""

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

print(mnist.data)
print(mnist.target)

print(np.shape(mnist.data))
print(np.shape(mnist.target))

N = 20000

from sklearn.model_selection import train_test_split

cut_x = np.resize(mnist.data, (N, 784))
cut_y = np.resize(mnist.target, N)

X_train, X_test, Y_train, Y_test = train_test_split(cut_x, cut_y, test_size=0.3, random_state=2)

print(np.shape(X_train))
print(np.shape(Y_train))

# KNN
print("KNN")
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
print("KNN ready")
predicted_knn = neigh.predict(X_test)
print("Knn predicted")

from sklearn import metrics

print("Accuracy KNN:", metrics.accuracy_score(Y_test, predicted_knn))

import knn

m_knn = knn.KnnClassification(n_neighbors=3)
m_knn.fit(X_train, Y_train)

print("myKnn predicted")

print("Accuracy my KNN:", metrics.accuracy_score(Y_test, m_knn.predict(X_test)))
