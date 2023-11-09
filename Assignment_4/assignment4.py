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
	result["best score"] = -1
	result["best_cluster"] = []

	max_range = range(2, min(cluster_range.stop, len(pokemon.index)))
	for n in max_range:
		pipeline.set_params(cluster__n_clusters = n)
		pipeline.fit(pokemon)
		labels = pipeline.predict(pokemon)
		score = silhouette_score(pipeline.named_steps["store_data"].data, labels)
		result["scores"].append(score)
		if(score > result["best score"]):
			result["best score"] = score
			result["best"] = n
			result["best_cluster"] = labels
	return result

# prints the clusters silhouette_scores in the specific output format
def print_cluster_scores(pok_type, result):
	print(pok_type)
	print("-----------")
	i = 0
	max_range = range(2, min(cluster_range.stop, len(type_instances[pok_type].index)))
	for n in max_range:
		print(str(n) + " clusters: " + str(result["scores"][i]))
		i = i + 1

	print("best number of clusters: " + str(result["best"]))
	print("best score: " + str(result["best score"]))
	print("")


# converts the pipeline output for a pokimon type into a dataframe for each cluster, then prints each dataframe
def print_cluster(pok_type, result):
	pok_info = type_instances[pok_type][["Name", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

	print(pok_type)
	print("-----")
	for n in np.unique(result["best_cluster"]):
		print("Cluster " + str(n))
		pok_in_cluster = pok_info[result["best_cluster"] == n]
		print(pok_in_cluster)

		print("Mean HP: " + str(pok_in_cluster["HP"].mean()))
		print("Mean Attack: " + str(pok_in_cluster["Attack"].mean()))
		print("Mean Defense: " + str(pok_in_cluster["Defense"].mean()))
		print("Mean Sp. Atk: " + str(pok_in_cluster["Sp. Atk"].mean()))
		print("Mean Sp. Def: " + str(pok_in_cluster["Sp. Def"].mean()))
		print("Mean Speed: " + str(pok_in_cluster["Speed"].mean()))
		
		print("")

	print("")

def main():
	type_results = {}
	for pok_type in types:
		type_results[pok_type] = cluster(pok_type)
		print_cluster_scores(pok_type, type_results[pok_type])

	# print the best clusters
	for pok_type in type_results:
		print_cluster(pok_type, type_results[pok_type])
main()