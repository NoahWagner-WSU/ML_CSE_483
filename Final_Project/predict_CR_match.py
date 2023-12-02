import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	# don't need to do anything
	def fit(self, xs, ys, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns]

data = pd.read_csv("data/cleaned_CR_data_Jan2.csv", nrows=100000)
xs = data.drop(columns = ["blue_wins"])
ys = data["blue_wins"]

grad_boost = None

with open("grad_boost.pkl", "rb") as f:
	grad_boost = pickle.load(f)

pred_y = grad_boost.predict(xs)

accuracy = accuracy_score(ys, pred_y)

print(accuracy)