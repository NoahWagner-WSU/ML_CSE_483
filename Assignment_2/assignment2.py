import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	# don't need to do anything
	def fit(self, xs, ys, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns].fillna(0)

# class ExpandColumns(BaseEstimator, TransformerMixin):
# 	def __init__(self, columns):
# 		self.columns = columns
# 	# don't need to do anything
# 	def fit(self, xs, ys, **params):
# 		return self
# 	# actually perform the selection
# 	def transform(self, xs):
# 		return xs.join(pd.get_dummies(xs[self.columns], dtype=float))

grid = {
	# "column_expand__columns" : [
	# 	["Kitchen Qual", "Neighborhood"]
	# ],
	"column_select__columns": [
		["Overall Qual"],
		["Overall Qual", "TotRms AbvGrd"],
		["Overall Qual", "TotRms AbvGrd", "Full Bath", "Garage Cars"],
		["Full Bath", "Garage Cars", "TotRms AbvGrd", "Fireplaces", "Year Built", "Overall Qual"],
		["Full Bath", "Garage Cars", "TotRms AbvGrd", "Fireplaces", "Year Built", "Overall Qual", "Kitchen Qual_Ex", "Kitchen Qual_Gd"],
		["Full Bath", "Garage Cars", "TotRms AbvGrd", "Fireplaces", "Year Built", "Overall Qual", "Kitchen Qual_Ex", "Kitchen Qual_Gd", "Neighborhood_NridgHt"]
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
		TransformedTargetRegressor(
			LinearRegression(n_jobs = -1),
			func = lambda y: np.power(y, 1/1.5),
			inverse_func = lambda y: np.power(y, 1.5)),
		TransformedTargetRegressor(
			LinearRegression(n_jobs = -1),
			func = lambda y: np.power(y, 1/2.5),
			inverse_func = lambda y: np.power(y, 2.5))
	]
}

steps = [
	# ("column_expand", ExpandColumns([])),
	("column_select", SelectColumns([])),
	("linear_regression", None),
]

pipe = Pipeline(steps)

search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)

data = pd.read_csv("AmesHousing.csv")

xs = data.drop(columns = ["SalePrice"])
xs = xs.join(pd.get_dummies(xs[["Kitchen Qual", "Neighborhood"]], dtype=float))
ys = data["SalePrice"]
train_x, test_x, train_y, test_y = train_test_split(xs, ys, train_size = 0.7)

search.fit(xs, ys)

print(search.best_score_)
print(search.best_params_)