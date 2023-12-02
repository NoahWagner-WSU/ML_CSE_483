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

xs = pd.read_csv("CR_Matches.csv")

grad_boost = None

with open("grad_boost.pkl", "rb") as f:
	grad_boost = pickle.load(f)

pred_y = grad_boost.predict(xs)

print(pred_y)