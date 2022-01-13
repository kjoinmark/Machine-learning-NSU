import matplotlib.pyplot as plt
import numpy as np

# Roc (кривая ошибок - графичекая характеристика качества бинарного классификатора,
# зависимость доли верных положительных классификаций от доли ложных положительных
# классификаций при варьировании порога решающего правила.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
iris_x, iris_y = load_iris(return_X_y=True)
X = iris_x
y = (iris_y == 1)
# plt.plot(X, y, 'o')
# plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

result = clf.fit(X_train, y_train)
y_predicted = result.predict_proba(X_test)[:, 1]


def precision(y_answ, y_true, threshold):
    y_predict = (y_answ >= threshold)
    tp = ((True == y_true) & (True == y_predict))
    fp = ((False == y_true) & (True == y_predict))
    if sum(tp) == 0 & sum(fp) == 0:
        return 1
    return sum(tp) / (sum(tp) + sum(fp))


def recall(y_answ, y_true, threshold):
    y_predict = (y_answ >= threshold)
    tp = ((True == y_true) & (True == y_predict))
    fn = ((True == y_true) & (False == y_predict))
    return sum(tp) / (sum(tp) + sum(fn))


def pr_curve(y_answ, y_true):
    x_curve = np.linspace(0, 1, 100*len(y_answ))
    x_curve = np.flip(x_curve)
    recall_curve = [recall(y_answ, y_true, i) for i in x_curve]
    precision_curve = [precision(y_answ, y_true, i) for i in x_curve]

    return recall_curve, precision_curve


r, p = pr_curve(y_predicted, y_test)

from sklearn import metrics

precision_, recall_, _ = metrics.precision_recall_curve(y_test, y_predicted)
pr_auc = metrics.auc(recall_, precision_)



m_auc = np.trapz(p, r)
f, ax = plt.subplots(2)
ax[0].plot(recall_, precision_, color='r', label='PR curve (area = %0.2f)' % pr_auc)
ax[1].plot(r, p, color='b', label='PR curve (area = %0.2f)' % m_auc)
ax[0].legend(loc="lower left")
ax[1].legend(loc="lower left")

plt.show()


def TPR(y_answ, y_true, threshold):
    y_predict = (y_answ >= threshold)
    tp = ((True == y_true) & (True == y_predict))
    fn = ((True == y_true) & (False == y_predict))
    return sum(tp) / (sum(tp) + sum(fn))


def FPR(y_answ, y_true, threshold):
    y_predict = (y_answ >= threshold)
    fp = ((False == y_true) & (True == y_predict))
    tn = ((False == y_true) & (False == y_predict))
    return sum(fp) / (sum(fp) + sum(tn))


def roc_curve(y_answ, y_true):
    x_curve = np.linspace(0, 1, len(y_answ)*100)
    x_curve = np.flip(x_curve)
    tpr_curve = [TPR(y_answ, y_true, i) for i in x_curve]
    fpr_curve = [FPR(y_answ, y_true, i) for i in x_curve]
    return fpr_curve, tpr_curve


fpr, tpr = roc_curve(y_predicted, y_test)
fpr_, tpr_, thresholds = metrics.roc_curve(y_test, y_predicted, pos_label=1)

roc_auc = metrics.auc(fpr_, tpr_)
m_roc_auc = np.trapz(tpr, fpr)

f, ax = plt.subplots(2)

ax[0].plot(fpr_, tpr_, color='r', label='PR curve (area = %0.2f)' % roc_auc)
ax[1].step(fpr, tpr, color='b', label='PR curve (area = %0.2f)' % m_roc_auc)
ax[0].legend(loc="lower right")
ax[1].legend(loc="lower right")

plt.show()


n = 50
x_ = np.random.randint(0,2,size = n)
x_p = np.full(n, 0.7)
x_ = (x_ == 1)

x_fpr, x_tpr, thresholds_ = metrics.roc_curve(x_, x_p, pos_label=1)

f, ax = plt.subplots()
ax.plot(x_fpr, x_tpr, color='b', label='PR curve (area = %0.2f)')
plt.show()