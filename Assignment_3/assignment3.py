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
		return xs[self.columns]

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

xs = data.drop(columns = ["SalePrice", "Neighborhood"])
xs.fillna(0, inplace=True)
xs = pd.get_dummies(xs, dtype=float)
xs = xs.select_dtypes(include='number')

ys = data["SalePrice"]

search.fit(xs, ys)

print(search.best_score_)
print(search.best_params_)