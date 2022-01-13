import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


def min_max_scaling(x):  # min-max scaling
    w, h = np.shape(x)
    X = x.copy()
    for i in range(h):
        _max = max(X[:, i])
        _min = min(X[:, i])
        X[:, i] = (X[:, i] - _min) / (_max - _min)
    return X


iris_x, iris_y = load_iris(return_X_y=True)
X = iris_x
#X = min_max_scaling(iris_x)
y = iris_y
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=2)


def mera(x):
    return np.power(sum((x) ** len(x)), 1 / len(x))


def k_epanechnicova(x, xi, h):
    r = mera(abs(x - xi)) / h
    return 3 / 4 * (1 - r ** 2)


def k_quadratic(x, xi, h):
    r = mera(abs(x - xi)) / h
    return 15 / 16 * (1 - r ** 2) ** 2


def k_triangle(x, xi, h):
    r = mera(abs(x - xi)) / h
    return 1 - r


def algorithm(x, x_train, y_train, gamma, kernel, h):
    weight = np.zeros(len(x_train))
    for i, xi in enumerate(x_train):
        weight[i] = kernel(x, xi, h) * gamma[i]
    res0 = sum((y_train == 0) * weight)
    res1 = sum((y_train == 1) * weight)
    res2 = sum((y_train == 2) * weight)
    ans = max(res0, res1, res2)
    if ans == res0:
        return 0
    if ans == res1:
        return 1
    if ans == res2:
        return 2


def train(x_train, y_train, kernel, h):
    length = len(x_train)
    gamma = np.zeros(length)
    while True:
        errors = 0
        for i in range(length):
            if y_train[i] != algorithm(x_train[i], x_train, y_train, gamma, kernel, h):
                gamma[i] += 1
                errors += 1
        if (errors / length < 0.1):
            break

    return gamma


H = 10

print("started")

gamma_t = train(X_train, Y_train, k_triangle, H)
print("triangle ready")

gamma_q = train(X_train, Y_train, k_quadratic, H)
print("quadratic ready")

gamma_e = train(X_train, Y_train, k_epanechnicova, H)
print("epanechnicova ready")

res_t = 0
res_q = 0
res_e = 0
l = len(X_test)

for i in range(l):
    res_t += algorithm(X_test[i], X_train, Y_train, gamma_t, k_triangle, H) == Y_test[i]
    res_q += algorithm(X_test[i], X_train, Y_train, gamma_q, k_quadratic, H) == Y_test[i]
    res_e += algorithm(X_test[i], X_train, Y_train, gamma_e, k_epanechnicova, H) == Y_test[i]

print(res_t / l)
print(res_q / l)
print(res_e / l)

y_t = ((Y_train == 0) * (gamma_t != 0), (Y_train == 1) * (gamma_t != 0), (Y_train == 2) * (gamma_t != 0))
y_q = ((Y_train == 0) * (gamma_q != 0), (Y_train == 1) * (gamma_q != 0), (Y_train == 2) * (gamma_q != 0))
y_e = ((Y_train == 0) * (gamma_e != 0), (Y_train == 1) * (gamma_e != 0), (Y_train == 2) * (gamma_e != 0))

f, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].plot(y_t[0] * X_train[:, i], y_t[0] * X_train[:, i], 'xr', label="first class")
    ax[i].plot(y_t[1] * X_train[:, i], y_t[1] * X_train[:, i], 'xb', label="second class")
    ax[i].plot(y_t[2] * X_train[:, i], y_t[2] * X_train[:, i], 'xg', label="third class")
    ax[i].legend(loc="lower right")
ax[0].set_title("Sepal length")
ax[1].set_title("Sepal width")
ax[2].set_title("Petal length")
ax[3].set_title("Petal width")
plt.show()

# y_t[0]*X_train[:,i] if y_t[0]>0 else float('nan')

np.savetxt(f"gamma_triangle.txt", gamma_t, newline="\n", fmt='%f ')
np.savetxt(f"gamma_quadratic.txt", gamma_q, newline="\n", fmt='%f ')
np.savetxt(f"gamma_epanechnicova.txt", gamma_e, newline="\n", fmt='%f ')
# print(sum(algorithm(X_test, X_test, Y_train, gamma_t, k_triangle, h)))
