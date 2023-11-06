import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

data = pd.read_csv("Pokemon.csv")

cluster_range = range(2, 15)

types = data["Type 1"].unique()
type_instances = {}

for typ in types:
	type_instances[typ] = data.loc[data["Type 1"] == typ]

print(type_instances)

# def main():

# def print_cluster_scores(type, silhouette_scores):

# def check_n_clusters(type):
	# return array of silhouette_score