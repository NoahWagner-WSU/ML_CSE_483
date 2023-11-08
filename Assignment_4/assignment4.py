import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	# don't need to do anything
	def fit(self, xs, ys = None, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns]

class Store_Data(BaseEstimator, TransformerMixin):
	def fit(self, xs, ys = None, **params):
		return self

	def transform(self, xs):
		self.data = xs
		return xs


data = pd.read_csv("Pokemon.csv")

cluster_range = range(2, 15)

types = data["Type 1"].unique()
type_instances = {}

type_clusters = {}

for typ in types:
	type_instances[typ] = data.loc[data["Type 1"] == typ]

steps = [
	("column_select", SelectColumns(["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])),
	("minmax", MinMaxScaler()),
	("store_data", Store_Data()),
	("cluster", KMeans(1, n_init = 10)),
]

pipeline = Pipeline(steps)

# try fitting the pipline for cluster__n_cluster in cluster_range
def cluster(pok_type):
	pokemon = type_instances[pok_type]
	result = {}
	result["scores"] = []
	result["best"] = 1
	result["best_cluster"] = []
	best_score = -1

	max_range = range(2, min(cluster_range.stop, len(pokemon.index)))
	for n in max_range:
		pipeline.set_params(cluster__n_clusters = n)
		pipeline.fit(pokemon)
		labels = pipeline.predict(pokemon)
		score = silhouette_score(pipeline.named_steps["store_data"].data, labels)
		result["scores"].append(score)
		if(score > best_score):
			best_score = score
			result["best"] = n
			result["best_cluster"] = labels
	return result

def main():
	type_results = {}
	for pok_type in types:
		type_results[pok_type] = cluster(pok_type)
		# print_cluster_scores(pok_type, type_results[pok_type].scores)

	print(type_results)
	# print the clusters

main()

# converts the pipeline output for a pokimon type into a dataframe for each cluster, then prints each dataframe
# print_cluster(type)

# prints the clusters silhouette_scores in the specific output format
# def print_cluster_scores(type, silhouette_scores):