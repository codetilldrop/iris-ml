import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idxs = [0,50,100]

# training data
train_target = np.delete(iris.target, test_idxs)
train_data = np.delete(iris.data, test_idxs, axis=0)

# testing the data with the 3 indexes removed
test_target = iris.target[test_idxs]
test_data = iris.data[test_idxs]

# training with decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))