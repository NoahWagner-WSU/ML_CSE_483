import numpy as np
import pandas as pd
import math
import sys

data = pd.read_csv("zoo.csv")
train = data.sample(frac = 0.7)
test = data.drop(train.index)

class_types = data["class_type"].unique()

# laplace smoothing variables
a = 0.01
D = 7

# gets the conditional probability of a feature given a class
# also applies laplace smoothing
# assumes feature is of the form [type, amount] to account for a different amount of legs
# the type is an index to the feature column
def smooth_cond_prob(feature, c_type):
	num = ((train["class_type"] == c_type) & (train.iloc[:, feature[0]] == feature[1])).sum() + a
	den = (train["class_type"] == c_type).sum() + a * D
	return num / den

# do log thing
# multiplies the conditional probabilities for all features given the class
# c_type is the class type, and instance is a pandas series of an instance (only features are used)
def naive_bayes(c_type, instance):
	log_sum = 0
	for feature in range(1, len(instance) - 1):
		log_sum += math.log(smooth_cond_prob([feature, instance[feature]], c_type), 2)
	log_sum += math.log((train["class_type"] == c_type).sum() / len(train.columns), 2)
	return 2**log_sum

# return class type and soft-max probability in an array or dictionary idk
def predict(instance):
	nb_results = []
	nb_total = 0
	for c_type in class_types:
		nb_results.append(naive_bayes(c_type, instance))
	nb_total = sum(nb_results)

	prediction = nb_results.index(max(nb_results))

	return [class_types[prediction], nb_results[prediction] / nb_total]

def main():
	predictions = []
	probabilities = []
	correct = []
	classes = test["class_type"]
	for row in range(0, len(test.index)):
		prediction = predict(test.iloc[row])
		predictions.append(prediction[0])
		probabilities.append(prediction[1])
		is_correct = (classes.iloc[row] == prediction[0])
		if is_correct:
			correct.append("CORRECT")
		else:
			correct.append("WRONG")
	test["predicted"] = predictions
	test["probability"] = probabilities
	test["correct?"] = correct

main()

test.to_csv(sys.stdout, sep=',', index=False)