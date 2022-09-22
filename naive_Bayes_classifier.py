from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import pandas as pd

print("we are going to go with the Gaussian version of Naive Bayes (for continuous input data)")
print("we will then make a function that does Naive Bayes for discrete probabilities from scratch (using and not using smoothing)")
print("given the feature inputs, it may be 'Naive' to assume all inputs are independent of each other, but it's a simplifying assumption")
print("there is also an implicit assumption that each feature input is equally important in predictive power")
print("now joint probabilities can just be multiplied together")
print("we could be working w/ categorical/discrete or continuous inputs.")

iris = load_iris()

print("load the data from sklearn.datasets")
X = iris.data
y = iris.target

print()
print("(row, column) shape of data features and label:")
print(X.shape)
print(y.shape)

print()
print("data type")
print(type(X))

print()
print("feature data:")
print(X)
print("target data:")
print(y)

print()
print("'train_test_split' the data")
print("'test_size' will be 20%, and 'random_state' to get back to split will be 42")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

print()
print("instantiate the 'GaussianNB' (naive Bayes) model...")
print("GNB assumes that the four inputs are independent.")
print("when coming up w/ prediction probabilities it multiplies their likelihoods together (without correlation terms).")
gnb = GaussianNB()

print()
print("fit the Gaussian Naive Bayes model...")
# Are conditional probabilities for target X feature_value just computed?
# skip 10 fold cross validation... is it even needed?
gnb.fit(X_train, y_train)

# test it
print()
print("use the fitted model to make a prediction on the hold-out, test set...")
y_pred = gnb.predict(X_test)

print()
print("accuracy details: predictions vs answers:")
compare = zip(y_pred, y_test)
for tup in compare:
    print(tup)

print()
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


print("##############Let's make a Naive Bayes classifier for categorical inputs################")
print("check: calculate prior & conditional probabilities, p(classification_i|feature_j=x_j) on our own (assuming discrete inputs)")
print("Bayes' Theorem: p(a|b) = p(b|a)*p(a)/p(b)")
print("Bayes' ML syntax: p(y|X) = p(X|y)*p(y)/p(X) = p(y|X_1, X_2... X_n) = p(X_1|y)*p(X_2|y)*...*p(X_n|y)*p(y)/p(X)")
print("The prediction probability is a distribution over the number of labels/categories")
print("Since each possible label is divided by p(X_1)*p(X_2)...p(X_n), we can skip the denominator and just scale it to sum to 1")
print("p(y) are the label class frequencies seen in y_train")
print("p(X_i|y) are filtered by y label column")

print("calculate distribution of feature values for each column")  # denominators, p(b), p(feature_j=x)")
col_val_dicts = []
for col in range(len(X_train[0])):
    unique, counts = np.unique(X_train[:, col], return_counts=True)
    distribution = dict(zip(unique, counts))
    col_val_dicts.append(distribution)

# test_example = [5.1, 3.5, 1.4, 0.2]
test_example = [4.7, 3.2, 1.3, 0.2]
test_example_prior_prob = []
test_example_conditional_prob = []  # conditioned on being in one of the categories

Xy_train = pd.DataFrame(X_train)
Xy_train["y"] = y_train

print("calculate prior probabilities for y, before we see a feature vector X")
prior_prob_y = {label: len(Xy_train[Xy_train["y"] == label])/len(Xy_train["y"]) for label in Xy_train["y"].unique()}
print(prior_prob_y)

print("create a dictionary(w/ keys=features) that return values = dictionary(w/ keys=feature_categories)...")
print("that returns a value = dictionary(w/ key=label_categories) which returns a prob, p(X_i|y=2")
cond_probs = {}
for i, feature in enumerate(Xy_train.columns):
    temp_dict = {}
    for cat in Xy_train[feature].unique():
        temp_dict2 = {}
        for label in Xy_train["y"].unique():
            temp_dict2[label] = len(Xy_train[(Xy_train[Xy_train[feature] == cat]) & (Xy_train[Xy_train["y"] == label])])/prior_prob_y[label]
        temp_dict[cat] = temp_dict2
    cond_probs[feature] = temp_dict



print("in case of 0 occurrences, add +1 to numerator, +2 to denominator")
for col, val in enumerate(test_example):
    print("create prior probabilities, p(classification_i)")
    if val in col_val_dicts[col]:
        test_example_prior_prob.append((col_val_dicts[col][val] + 0)/(len(X_train) + 0))
    else:
        test_example_prior_prob.append(0/(len(X_train) + 0))
    # if val in col_val_freq_dicts[col]:
    #     test_example_prior_prob.append((col_val_freq_dicts[col][val] + 0)/(len(X_train) + 0))
    # else:
    #     test_example_prior_prob.append(0/(len(X_train) + 0))

    print("calculate conditional probabilities, p(classification_i|feature_j=x)")
    temp_cond = []
    for prediction in [0, 1, 2]:
        denominator = len(Xy_train[Xy_train["y"] == prediction]) + 0
        numerator = len(Xy_train[(Xy_train["y"] == prediction) & (Xy_train.columns[col] == val)]) + 0
        temp_cond.append(numerator/denominator)

    test_example_conditional_prob.append(temp_cond)


print("prior probabilities without smoothing:")
print(test_example_prior_prob)
print("conditional probabilities without smoothing:")
print(test_example_conditional_prob)

print("shape", np.array(test_example).shape)
np_array_test_example = np.array(test_example).reshape(1, -1)
print("reshape", np_array_test_example.shape)
print("prediction y = argmax_y(prediction_distribution)")
predict_proba = gnb.predict_proba(np_array_test_example)

print("conditional probabilities based on labels")
print(predict_proba)


