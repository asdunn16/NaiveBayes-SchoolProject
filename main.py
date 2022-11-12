from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# load dataset
iris = datasets.load_iris()

# print feature labels
print('Features:', iris.feature_names)

# print class labels
print('Classes:', iris.target_names)

# split dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.25, random_state=1)

# create model
model = GaussianNB()

# prediction from test set
y_predict = model.fit(X_train, y_train).predict(X_test)

# class count
print('-------------------------------------------------------')
print('Number of training samples in each class:')
print(iris.target_names[0] + ':', model.class_count_[0])
print(iris.target_names[1] + ':', model.class_count_[1])
print(iris.target_names[2] + ':', model.class_count_[2])
print('Total number of training samples:', model.class_count_[0] + model.class_count_[1] + model.class_count_[2])

# probability of each class
print('-------------------------------------------------------')
print('Probability of each class from training samples:')
print(iris.target_names[0] + ':', (model.class_prior_[0] * 100))
print(iris.target_names[1] + ':', (model.class_prior_[1] * 100))
print(iris.target_names[2] + ':', (model.class_prior_[2] * 100))
print('-------------------------------------------------------')

# number mislabeled from test set
print('Number of mislabeled points out of a total', X_test.shape[0], 'points:', (y_test != y_predict).sum())

# accuracy of classification
print('Accuracy of model:',  (model.score(X_test, y_test) * 100))
