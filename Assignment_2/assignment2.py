import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__( self, columns ):
		self.columns = columns
	# don't need to do anything
	def fit(self, xs, ys, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns]

grid = { 
	"column_select__columns": [
		["Gr Liv Area"],
		["Gr Liv Area", "Overall Qual"],
		["Gr Liv Area", "Overall Qual", "Year Built"],
	],
	"linear_regression": [
		LinearRegression(n_jobs = -1), # no transformation
		TransformedTargetRegressor(
			LinearRegression(n_jobs = -1),
			func = np.sqrt,
			inverse_func = np.square ),
		TransformedTargetRegressor(
			LinearRegression(n_jobs = -1),
			func = np.cbrt,
			inverse_func = lambda y: np.power(y, 3)),
		TransformedTargetRegressor(
			LinearRegression(n_jobs = -1),
			func = np.log,
			inverse_func = np.exp),
	]
}

steps = [
	("column_select", SelectColumns([])),
	("linear_regression", None),
]

pipe = Pipeline(steps)

search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)

data = pd.read_csv("AmesHousing.csv")

xs = data.drop( columns = ["SalePrice"])
ys = data["SalePrice"]
train_x, test_x, train_y, test_y = train_test_split(xs, ys, train_size = 0.7)

search.fit(xs, ys)

print(search.best_score_)
print(search.best_params_)