import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv("AmesHousing.csv")

xs = data.drop(columns = ["SalePrice", "Neighborhood"])
xs.fillna(0, inplace=True)
xs = pd.get_dummies(xs, dtype=float)
xs = xs.select_dtypes(include='number')

ys = data["SalePrice"]

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	# don't need to do anything
	def fit(self, xs, ys, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns]

grid_lin_reg = {
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
	"predictor": [
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

grid_dec_tree = {
	"column_select__columns": [
		[
			"Full Bath",
			"Garage Cars",
			"TotRms AbvGrd",
			"Total Bsmt SF",
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
			"Lot Shape_IR1",
			"Year Remod/Add",
			"Exter Qual_TA",
			"Bsmt Exposure_Gd",
			"Gr Liv Area",
		],
	],
	"predictor": [
		DecisionTreeRegressor()
	],
	"predictor__max_depth": [6, 7, 8, 9, 10],
	"predictor__max_features": [4, 6, 8, 10, 12, 14, 16],
}

grid_rand_forest = {
	"column_select__columns": [
		[
			"Full Bath",
			"Garage Cars",
			"TotRms AbvGrd",
			"Total Bsmt SF",
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
			"Lot Shape_IR1",
			"Year Remod/Add",
			"Exter Qual_TA",
			"Bsmt Exposure_Gd",
			"Gr Liv Area",
		],
	],
	"predictor": [
		RandomForestRegressor(n_jobs=-1)
	],
	"predictor__max_depth": [6, 7, 8, 9, 10],
	"predictor__max_features": [2, 3, 4, 6],
}

grid_grad_boost = {
	"column_select__columns": [
		[
			"Full Bath",
			"Garage Cars",
			"TotRms AbvGrd",
			"Total Bsmt SF",
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
			"Lot Shape_IR1",
			"Year Remod/Add",
			"Exter Qual_TA",
			"Bsmt Exposure_Gd",
			"Gr Liv Area",
		],
	],
	"predictor": [
		GradientBoostingRegressor()
	],
	"predictor__max_depth": [2, 3, 4, 5, 6, 7],
	"predictor__max_features": [2, 3, 4, 6],
	"predictor__learning_rate": [0.05, 0.1, 0.2],
}

steps = [
	("column_select", SelectColumns([])),
	("predictor", None),
]

pipe = Pipeline(steps)

search_lin_reg = GridSearchCV(pipe, grid_lin_reg, scoring = 'r2', n_jobs = -1)

search_dec_tree = GridSearchCV(pipe, grid_dec_tree, scoring = 'r2', n_jobs = -1)

search_rand_forest = GridSearchCV(pipe, grid_rand_forest, scoring = 'r2', n_jobs = -1)

search_grad_boost = GridSearchCV(pipe, grid_grad_boost, scoring = 'r2', n_jobs = -1)

search_lin_reg.fit(xs, ys)

print("Linear Regression:")
print("R-squared:", search_lin_reg.best_score_)
print("Best params:", search_lin_reg.best_params_)

search_dec_tree.fit(xs, ys)

print("Decision Tree:")
print("R-squared:", search_dec_tree.best_score_)
print("Best params:", search_dec_tree.best_params_)

search_rand_forest.fit(xs, ys)

print("Random Forest:")
print("R-squared:", search_rand_forest.best_score_)
print("Best params:", search_rand_forest.best_params_)

search_grad_boost.fit(xs, ys)

print("Gradient Boosting:")
print("R-squared:", search_grad_boost.best_score_)
print("Best params:", search_grad_boost.best_params_)