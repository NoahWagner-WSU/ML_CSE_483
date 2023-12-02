import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

#TODO: 
# ignore all gamemodes except 72000201
# possibly take into account trophy count
# add one more step to pipeline to meet requirements
# have user be able to input decks, and our model predict who will win

instance_count = 10000

data = pd.read_csv("data/cleaned_CR_data_Jan1.csv", nrows=instance_count)

xs = data.drop(columns = ["blue_wins"])
ys = data["blue_wins"]

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	# don't need to do anything
	def fit(self, xs, ys, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns]

select_cols = [
	"blue.card1.id",
	"blue.card1.level",
	"blue.card2.id",
	"blue.card2.level",
	"blue.card3.id",
	"blue.card3.level",
	"blue.card4.id",
	"blue.card4.level",
	"blue.card5.id",
	"blue.card5.level",
	"blue.card6.id",
	"blue.card6.level",
	"blue.card7.id",
	"blue.card7.level",
	"blue.card8.id",
	"blue.card8.level",
	"blue.totalcard.level",
	"blue.elixir.average",
	"red.card1.id",
	"red.card1.level",
	"red.card2.id",
	"red.card2.level",
	"red.card3.id",
	"red.card3.level",
	"red.card4.id",
	"red.card4.level",
	"red.card5.id",
	"red.card5.level",
	"red.card6.id",
	"red.card6.level",
	"red.card7.id",
	"red.card7.level",
	"red.card8.id",
	"red.card8.level",
	"red.totalcard.level",
	"red.elixir.average",
]

steps = [
	("column_select", SelectColumns(select_cols)),
	("scale", MinMaxScaler()),
	("predictor", None)
]

pipe = Pipeline(steps)

grid_rand_forest = {
	"predictor": [
		RandomForestClassifier(n_jobs=-1)
	],
	"predictor__max_depth": [20, 21, 22, 23],
	"predictor__max_features": ["sqrt", 12, 13, 14, 15, 16],
}

grid_grad_boost = {
	"predictor": [
		GradientBoostingClassifier()
	],
	"predictor__max_depth": [12, 13, 14, 15],
	"predictor__max_features": [5, 6, 7, 8, 9, 10],
	"predictor__learning_rate": [0.025, 0.05, 0.075],
}

search_rand_forest = GridSearchCV(pipe, grid_rand_forest, scoring = "accuracy", n_jobs=-1)

search_grad_boost = GridSearchCV(pipe, grid_grad_boost, scoring = "accuracy", n_jobs=-1)

# search_rand_forest.fit(xs, ys)

# print("Random Forest:")
# print("Accuracy:", search_rand_forest.best_score_)
# print("Best params:", search_rand_forest.best_params_)

search_grad_boost.fit(xs, ys)

print("Gradient Boosting:")
print("Accuracy:", search_grad_boost.best_score_)
print("Best params:", search_grad_boost.best_params_)

# save the model
with open("grad_boost.pkl", "wb") as f:
	pickle.dump(search_grad_boost.best_estimator_, f)