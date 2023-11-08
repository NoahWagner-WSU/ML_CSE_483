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
	def fit(self, xs, ys, **params):
		return self
	# actually perform the selection
	def transform(self, xs):
		return xs[self.columns]

data = pd.read_csv("Pokemon.csv")

cluster_range = range(2, 15)

types = data["Type 1"].unique()
type_instances = {}

type_clusters = {}

for typ in types:
	type_instances[typ] = data.loc[data["Type 1"] == typ]

steps = {
	"column_select": SelectColumns(["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]),
	"minmax": MinMaxScaler(),
	"cluster": KMeans(),
}

pipeline = Pipeline(steps)

def main():
	for pok_type in types:
		print_cluster_scores(pok_type, cluster(pok_type))

	# print the clusters

# converts the pipeline output for a pokimon type into a dataframe for each cluster, then prints each dataframe
# print_cluster(type)

# prints the clusters silhouette_scores in the specific output format
# def print_cluster_scores(type, silhouette_scores):

# try fitting the pipline for cluster__n_cluster in cluster_range
# store the best silhouette clusters in type_clusters[type], so type_clusters[type] = pipeline.predict(x)
# def cluster(type):
	# return array of silhouette_scores for each n_cluster in cluster_range, whose index i cooresponds to the ith number in cluster range