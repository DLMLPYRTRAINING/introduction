# KNN graph display---------------------
from scipy.sparse import coo_matrix
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier,kneighbors_graph
from sklearn.neighbors import kneighbors_graph

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",header=None)

df.columns = ['ColA','ColB','ColC','ColD','ColE']
df.rename(columns={'ColA': 'cn', 'ColB': 'lw', 'ColC': 'ld', 'ColD': 'rw', 'ColE': 'rd'},inplace=True)

X = df[['lw','ld','rw','rd']][:-2].values
Y = df["cn"][:-2].values

x_test = df[['lw','ld','rw','rd']][-2:].values
y_test = df["cn"][-2:].values

print("KNN----------------")
model1 = KNeighborsClassifier(n_neighbors=3)

model1.fit(X,Y.ravel())

score = model1.score(X,Y.ravel())
print(score)

print(x_test, y_test)
print(model1.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[5,5,5,3]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[3,5,5,3]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[3,5,5,5]]))



# X = [[0, 1], [3, 5], [1, 2], [5, 9], [20, 21], [23,30], [26, 22]]
A = kneighbors_graph(X, 3, mode='connectivity', include_self=True)
print(A.toarray())
ax = plot_coo_matrix(A)
ax.figure.show()
time.sleep(15)
