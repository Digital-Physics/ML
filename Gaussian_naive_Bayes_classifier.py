from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import pandas as pd
import math

print("we are going to go with the Gaussian (for continuous input data) version of a Naive Bayes classifier")
print("look into kernel density estimates... for estimating the pdf of functions based on local data")
print("if our inputs were 0/1 we'd do a BernoulliNB and if we had n-categorical inputs we could use a MultinomialNB classifier")
print("we will first work with continuous inputs using sklearn.naive_bayes.GaussianNB")
print("we will then make a function that does Naive Bayes for discrete probabilities from scratch (using Laplace smoothing?)")
print("it may be 'naive' to assume all inputs in your data are independent of each other, but it's a simplifying assumption...")
print("that allows joint probabilities to be created by simply multiplying different (conditional and prior) probabilities together...")
print("it may also be 'naive' to implicitly treat all of our inputs the same, and not weight feature importance")

iris = load_iris()

print()
print("load the data from sklearn.datasets")
X = iris.data
y = iris.target

print()
print("(row, column) shape of data features and label:")
print(X.shape)
print(y.shape)

print()
print("data type should be a numpy array")
print(type(X))

print()
print("feature data:")
print(X)
print("target data:")
print(y)

print()
print("'train_test_split' the data")
print("'test_size' will be 20%, and 'random_state' is an int we'll use to get back to the same split. sci-fi fans like 42 as a seed/state.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

print()
print("instantiate the 'GaussianNB' (naive Bayes) model...")
print("GNB assumes that the four inputs in our Iris data set (ie. sepal length, sepal width, petal length, petal width) are independent.")
gnb = GaussianNB()

print()
print("fit the Gaussian Naive Bayes model... behind the scenes mu(mean) and sigma(std dev) are being computed")
gnb.fit(X_train, y_train)

# skip 10-fold cross validation...

# test it
print()
print("use the fitted model to make a prediction on the hold-out, test set...")
y_pred = gnb.predict(X_test)

print()
print("zip together two vectors to look at accuracy details on the test_set - predictions vs answers:")
compare = zip(y_pred, y_test)
for tup in compare:
    print(tup)

print()
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


print("########################################################################################")
print("########################################################################################")
print("##############Let's make a Naive Bayes classifier for categorical inputs################")
print()
print("calculate 'Priors' (prob p(classification=label), p(y)) & (conditional) 'Likelihoods' (probs p(feature_j=x_j|classification=i)")
print("to make a quick and fast discrete model, we use the same exact Iris dataset but round inputs to integers, .rint(), to discretize it")
discrete_X_train = np.rint(X_train)
print(discrete_X_train)

print("Bayes' Theorem: p(a|b) = p(b|a)*p(a)/p(b)")
print("Think of the overlap in two Venn diagram circles a and b: p(a and b)/p(b)")
print("Side Note: Duh's Theorem: p(a|b) = p(a|b)*p(b)/p(b)")
print("Bayes' w/ ML variable: p(y|X) = p(X|y)*p(y)/p(X) = p(y|X_1, X_2... X_n) = p(X_1|y)*p(X_2|y)*...*p(X_n|y)*p(y)/p(X)")
print("p(X) = p(x_1|y)*p(x_2|y)...*p(x_n|y)")
print("The prediction probability given an input vector X, p(y|X), is a distribution over the number of label categories")
print("Since each label likelihood is divided by p(X) = p(X_1)*p(X_2)...p(X_n)...")
print("we can skip the proportional denominator in the calculation if we want and just scale conditional probs to sum to 1")
print("p(y) are the label class frequencies seen in y_train, which seem like reasonable priors without more information(i.e. X)")
print("p(X_i=category_i|y=label_j) are conditional on different outcomes in the training set (think filtering y column in excel)")
print("in the event they are 0, we can use something like Laplace smoothing to avoid giving no likelihood to certain labels")
print("if we were predicting spam, just because training didn't have any spam emails w/ the word 'esoteric' doesn't mean no chance of spam")


# test_example = [5.1, 3.5, 1.4, 0.2]
# test_example = [4.7, 3.2, 1.3, 0.2]
print("create a test example")
test_example = [5, 3, 1, 0]
test_example_prior_prob_P_y = []  # distribution over possible label categories
test_example_conditional_prob = []  # conditioned on being in one of the categories

discrete_Xy_train = pd.DataFrame(discrete_X_train)
discrete_Xy_train["y"] = y_train


class MultinomialNB:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        print("self.data", self.data)
        self.labels = self.data["y"].unique()
        self.features = self.data.drop(["y"], axis=1).columns
        print("self.feature", self.features)
        self.prior_prob_y = None
        self.cond_prob_X_given_y = None
        self.train()

    def train(self) -> None:
        print("store priors for each label in a dictionary/hash table using a 'dictionary comprehension'")
        print("key: value = label(str): prior_probability(float)")
        self.create_prior_prob_dict()  # for labels, p(y=j)
        self.create_conditional_prob_dict()  # for feature values, p(X=i|y=j)    used in Bayes' formula

    def create_prior_prob_dict(self) -> None:
        print("store priors in a dictionary/hash table using a 'dictionary comprehension'")
        print("key: value = label (int 0,1,2): prior_probability (float)")
        self.prior_prob_y = {label: len(self.data[self.data["y"] == label]) / len(self.data["y"]) for label in self.labels}

    def create_conditional_prob_dict(self) -> None:
        print("create a 3-layer nested dictionary to hold our conditional probabilities e.g. p(x_i=5|y=2)")
        print("{feature_col: feature_vals: label_vals: conditional_prob}")
        self.cond_prob_X_given_y = {}
        for feature in self.features:
            temp_dict = {}
            for cat in self.data[feature].unique():
                temp_dict2 = {}
                for label in self.labels:
                    # Laplace smoothing adds 1 to numerator and 2 to denominator
                    numerator_cat_occurences = len(
                        discrete_Xy_train[(discrete_Xy_train[feature] == cat) & (discrete_Xy_train["y"] == label)]) + 1
                    denominator_total_label = len(discrete_Xy_train[discrete_Xy_train["y"] == label]) + 2
                    temp_dict2[label] = numerator_cat_occurences / denominator_total_label
                temp_dict[cat] = temp_dict2
            self.cond_prob_X_given_y[feature] = temp_dict
        print("cond prob X given y (feature_col: feature_vals: label_vals: conditional_prob):")
        print(self.cond_prob_X_given_y)

    def predict(self, flower_x: list[str]) -> dict:
        """Use Naive Bayes' p(y|X = X_1, X_2, ... X_n) = p(X|y)*p(y)/p(X)
        = p(X_1|y)*p(X_2|y)*...p(X_n|y)*p(y)/p(X)"""
        loglikelihoods = {}

        for label in self.labels:
            # we can take the log of the probabilities (and add instead of multiply) because the log is convex
            # the highest probability will now have the least negative number, and therefore argmax will still make the same label choice
            # this will help prevent us from losing significant digits, which can happen when multiplying many numbers < 0 together
            numerator = math.log(self.prior_prob_y[label])
            # denominator = 0 # since the denominator is constant p(X) across all posteriors, we can ignore it. can scale to sum to 1.
            for feat_col, x_input in zip(self.features, flower_x):
                if x_input in self.cond_prob_X_given_y[feat_col]:
                    # += log(probs) instead of *= probs
                    numerator += math.log(self.cond_prob_X_given_y[feat_col][x_input][label])
                else:
                    numerator += math.log(0.5)

            loglikelihoods[label] = numerator

        print("posterior log-likelihoods before normalizing to sum to 1")
        print(loglikelihoods)
        return loglikelihoods


model = MultinomialNB(discrete_Xy_train)
print(model.predict(test_example))
