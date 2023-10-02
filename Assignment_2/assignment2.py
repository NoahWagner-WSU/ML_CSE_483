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

class TransformColumns(BaseEstimator, TransformerMixin):
	def __init__(self, func):
		self.func = func
	# don't need to do anything
	def fit(self, xs, ys, **params):
		return self
	# actually perform the transformation
	def transform(self, xs):
		return xs.apply(self.func)

grid = {
	"column_select__columns": [
		[
			"Full Bath", 
			"Garage Cars", 
			"TotRms AbvGrd", 
			"Fireplaces", 
			"Overall Qual", 
			"Kitchen Qual_Ex", 
			"Kitchen Qual_Gd", 
			"Neighborhood_NridgHt", 
			"Neighborhood_NoRidge",
			"Neighborhood_StoneBr",
			"Neighborhood_OldTown",
			"Garage Type_Attchd",
			"BsmtFin Type 1_GLQ",
			"Wood Deck SF",
			"Mas Vnr Area",
			"Screen Porch",
			"MS SubClass",
			"Lot Area",
			"Year Remod/Add",
			"Exter Qual_TA",
			"Bsmt Exposure_Gd",
		],
	],
	"column_transform__func": [
		lambda y: y,
		np.sqrt,
		np.square,
		np.cbrt,
		lambda y: np.power(y, 3),
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
	("column_transform", TransformColumns(None)),
	("linear_regression", None),
]

pipe = Pipeline(steps)

search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)

data = pd.read_csv("AmesHousing.csv")

xs = data.drop(columns = ["SalePrice"])
xs = pd.get_dummies(xs, dtype=float)
ys = data["SalePrice"]

search.fit(xs, ys)

print(search.best_score_)
print(search.best_params_)