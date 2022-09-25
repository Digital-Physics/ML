# Confusion Matrix:
# TP FP
# TN FN


# accuracy = correct_guesses/total_guesses = (TP + TN)/(TP + TN + FP + FN)

# but accuracy doesn’t tell the complete picture; what if the dataset was extremely unbalanced? Just never predict rare event.


# sensitivity = pos pred. value = TP/(TP + FN)

# sensitivity is good to focus on if you definitely want to catch stuff, like cancer… notice how False Positives don’t hurt the metric

# the denominator is just Actually True column… so like all of the cancer cases

# our model would be very “sensitive” and go off a lot… more False Positives, less False Negatives


# specificity = TN/(TN + FP)

# our model would not be “sensitive” and won’t go off much… less False Positives, more False Negatives


# precision = TP/(TP + FP)

# a test can cheat and just go off rarely but accurately and get a precision of 100%


# F1 score = harmonic mean = 2*(sensitivity*precision)/(sensitivity + precision)

# this balances the need to go off just to be safe (high sensitivity) w/ a model that only answers when it is sure (precision)


# there is a usually a sensitivity-specificity tradeoff… which is worse, False Positives or False Negatives?

# the ROC curve relates to this… how quickly can you get your sensitivity up without incurring more False Positives?

# ROC curve: y axis = sensitivity, x axis = 1-specificity

# A good model will have a higher AUC.

# A good rule of thumb is to pick a threshold on the ROC curve that is the maximum distance from the diagonal line


# Laplace Smoothing to avoid 0 probabilities in discrete input Naïve Bayes:

# Add epsilon +1 in numerator, +2 in denominator so nothing is impossible

# https://en.wikipedia.org/wiki/Additive_smoothing


# K-fold Cross Validation

# Tradeoff: 10-Fold takes 10 times as long, although less of a chance we split our data in a less than optimal way… resulting in less performance

# Average performance


# #######################


# Bernoulli model vs. Multinomial model

# yes_no_text_vector vs. count_text_vector

# vs. term_frequency_vector


# TF-IDF score: (TF*IDF)

# Term Frequency – Inverse Document Frequency

# TF: “travel” appears 10 times out of 200 words = 5%

# IDF: log(#documents/(#documents where word appears)): “travel” appears in every article! = log(1) = 0

# This metric tries to measure whether that word (e.g. “travel”) is a relevant word to classify on

# If “travel” was in every document, TF*IDF = 5% * 0 = 0 => “travel” isn’t informative

# This is analogous to removing “stop words” after “tokenizing text”

# TF-IDF number can be used to adjust the “travel” index in the term_freq_vector