import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier,kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",header=None)
# print(df)

df.columns = ['ColA','ColB','ColC','ColD','ColE']
df.rename(columns={'ColA': 'cn', 'ColB': 'lw', 'ColC': 'ld', 'ColD': 'rw', 'ColE': 'rd'},inplace=True)
# this should work with updated column name from the line above
# df = df.rename(columns={'cn':'C11','lw':'C22','ld':'C33','rw':'C44','rd':'C55'})
# print(df)

# no need to convert to numeric category here
# df.loc[df["cn"] == "L", "cn"] = 0
# df.loc[df["cn"] == "B", "cn"] = 1
# df.loc[df["cn"] == "R", "cn"] = 2

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

print("RFC----------------")

model2 = RandomForestClassifier()

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model2.predict([[5,5,5,3]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model2.predict([[3,5,5,3]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model2.predict([[3,5,5,5]]))

print("LOGR---------------")

model2 = LogisticRegressionCV(cv=10,multi_class = "multinomial")

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model2.predict([[5,5,5,3]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model2.predict([[3,5,5,3]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model2.predict([[3,5,5,5]]))

print("SVC----------------")

model2 = SVC()

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model2.predict([[5,5,5,3]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model2.predict([[3,5,5,3]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model2.predict([[3,5,5,5]]))

# so what's going wrong?:
# The challenge is a classification: if the row of data belongs to class B, L or R
# it should be able to properly categorise the data into either left, right or both
#  what is happening is every model is showing a very good score when tested with internal data
# but almost all the time is guessing the output wrong when it comes to test data.

# Also the models works fine(so many different model won't generate same pattern of error if the issue was with model and not with data)
# with this dataset as long and the test data is within the dataset and not outside.
# the complete training happens with data between 1,1,1,1 and 5,5,5,4 and the test data is just outside
#  which is 5,5,5,5 so the model don't understand what to do. This is a typical example of model overfitting or
# target imbalance.

# This occurs usually when either the dataset is very small which in this case it is but on top of it what makes the
# matter worst is that the example for B class is about only 40 rows where L and R is around 230 rows. So there aren't
# enough data for the B class so that the model learns well to generalise so that it can handle even unseen data

# There are 2 things that can help us here, using feature engineering come up with some feature that makes the
# separation easly for the models or come up with more sample data

# Here as example: i am reducing the number of features from 4 to 2 by multiplying lw & ld to create a new feature column 1
#  and multiplying rw & rd to generate new feature column 2. This will make the seperation of the data or relation between
#  these columns much much clear to the model that if new feature 1 < new feature 2 then it's L, if
# new feature 1 = new feature 2 then it's B or new feature 1 > new feature 2 then it's R.
# the models will still suffer because of the lack of many examples for the B category but hopefully the results
#  will improve.


print("Doing some feature engineering ======================================")

df['newcol1'] = df['lw']*df['ld']
df['newcol2'] = df['rw']*df['rd']

X = df[['newcol1','newcol2']][:-2].values
Y = df["cn"][:-2].values

x_test = df[['newcol1','newcol2']][-2:].values
y_test = df["cn"][-2:].values

print("KNN----------------")
model1 = KNeighborsClassifier(n_neighbors=3)

model1.fit(X,Y.ravel())

score = model1.score(X,Y.ravel())
print(score)

print(x_test, y_test)
print(model1.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[(5*5),(5*3)]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[(3*5),(5*3)]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[(3*5),(5*5)]]))


print("RFC----------------")

model2 = RandomForestClassifier()

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[(5*5),(5*3)]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[(3*5),(5*3)]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[(3*5),(5*5)]]))


print("LOGR---------------")

model2 = LogisticRegressionCV(cv=10,multi_class = "multinomial")

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[(5*5),(5*3)]])) # works fine
print(model1.predict([[(1.5*5),(1.5*4)]])) # works fine
print(model1.predict([[(20*15),(22*5)]])) # works fine

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[(3*5),(5*3)]])) # works fine
print(model1.predict([[(2.2*3),(3.3*2)]])) # works fine
print(model1.predict([[(6*6),(6*6)]])) # doesn't work
print(model1.predict([[(3*20),(20*3)]])) # doesn't work

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[(3*5),(5*5)]])) # works fine
print(model1.predict([[(5.1*1.5),(3.5*2.9)]]))  # works fine
print(model1.predict([[(31*5),(20*5)]])) # works fine

print("SVC----------------")

model2 = SVC()

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[(5*5),(5*3)]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[(3*5),(5*3)]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[(3*5),(5*5)]]))

# So, when you run this part of the code, you will notice logistic regressor classifier is able to
#  pretty much identify any kind of data you can provide within a certain limit. it is able to handle decimals,
# and a bit large numbers based on the new features we have engineered.

# The Challange is feature engineering if you google about you will find what i am going to say as i have been
# told the same by many data scientist that it is an art. there is no mathametical formula why i multiplied these columns
#  which helped the model to make better prediction(remember better prediction doesn't mean 100% accuracy.
# in the world of data science if your model has 100% accuracy, it's guranteed that your model has something wrong with it.
# actually anything with more than 80~85% accuracy is considered very good and people fight over 0.1% of
# accuracy improvement.) It is an art of seeing how best we can represent the data so that the model can find a
# pattern in it. and this will only come from trial and failures.

# So why logistic regression is winning here: Look at the data, it's a classification problem, true but it has a upword growing trend
# the data in X starts with 1,1,1,1 and ends at 5,5,5,3 so it has a direction to follow even though the conclusion we are trying to
# derive is classification. this is why logistic regresion which tries to plot a progressive curve on data can predict
# better than others right our of the box in this case. It  may not be the same in case of another dataset where there is not progress or
# movement trend in the data.

# what more we can do to improve our accuracy? 
# we can surely do more imaginative feature engineering, but we can also increase examples
# of B label data from just 40 rows to somewhere about 240 to compete with other labels in the dataset. this should increase
# the models idea about how to handle data when ld*lw == rd*rw.


# -----------------------------------------------------------------------------------------------------------
#
# The output in my system looks like this:
#
# KNN----------------
# 0.9213483146067416
# [[5 5 5 4]
#  [5 5 5 5]] ['L' 'B']
# ['B' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['B']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# RFC----------------
# 0.9951845906902087
# [[5 5 5 4]
#  [5 5 5 5]] ['L' 'B']
# ['B' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['B']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# LOGR---------------
# 0.9052969502407705
# [[5 5 5 4]
#  [5 5 5 5]] ['L' 'B']
# ['L' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['R']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# SVC----------------
# 0.9181380417335474
# [[5 5 5 4]
#  [5 5 5 5]] ['L' 'B']
# ['L' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['L']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# Doing some feature engineering ======================================
# KNN----------------
# 0.9967897271268058
# [[25 20]
#  [25 25]] ['L' 'B']
# ['L' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['B']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# RFC----------------
# 1.0
# [[25 20]
#  [25 25]] ['L' 'B']
# ['L' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['B']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# LOGR---------------
# 1.0
# [[25 20]
#  [25 25]] ['L' 'B']
# ['L' 'B']
# This should predict : left-L -> 5,5,5,3
# ['L']
# ['L']
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['B']
# ['B']
# ['R']
# ['R']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# ['R']
# ['R']
# SVC----------------
# 0.9983948635634029
# [[25 20]
#  [25 25]] ['L' 'B']
# ['L' 'R']
# This should predict : left-L -> 5,5,5,3
# ['L']
# This should predict : Both-B -> 3,5,5,3
# ['B']
# This should predict : Right-R -> 3,5,5,5
# ['R']
# 
# Process finished with exit code 0
