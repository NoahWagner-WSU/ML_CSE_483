import pandas as pd
import numpy as np
import random

instance_count = 1000

features = [
	"winner.card1.id",
	"winner.card1.level",
	"winner.card2.id",
	"winner.card2.level",
	"winner.card3.id",
	"winner.card3.level",
	"winner.card4.id",
	"winner.card4.level",
	"winner.card5.id",
	"winner.card5.level",
	"winner.card6.id",
	"winner.card6.level",
	"winner.card7.id",
	"winner.card7.level",
	"winner.card8.id",
	"winner.card8.level",
	"winner.totalcard.level",
	"winner.elixir.average",
	"loser.card1.id",
	"loser.card1.level",
	"loser.card2.id",
	"loser.card2.level",
	"loser.card3.id",
	"loser.card3.level",
	"loser.card4.id",
	"loser.card4.level",
	"loser.card5.id",
	"loser.card5.level",
	"loser.card6.id",
	"loser.card6.level",
	"loser.card7.id",
	"loser.card7.level",
	"loser.card8.id",
	"loser.card8.level",
	"loser.totalcard.level",
	"loser.elixir.average",
]

# consider only loading the winner.cards.list and loser.cards.list instead of the individual cards
# possibly account for gamemode, we only want to load normal games

# write a function to confirm if there are duplicate cards just to double check

data = pd.read_csv("BattlesStaging_01012021_WL_tagged.csv", usecols=features, nrows=instance_count)

data["blue_wins"] = 1

# doesn't work lol
def swap_halves(row):
	if(random.random() > 0.5):
		mid = len(row) // 2
		row[-1] = 0
		swapped = np.concatenate([row[mid:-1], row[:mid]])
		swapped = np.append(swapped, row[-1])
		return pd.Series(swapped, index=row.index)
	else:
		return row

data = data.apply(swap_halves, axis=1)

# convert features to every "winner" replaced with "blue", and "loser" with "red"
for i in range(0, len(features)):
	features[i] = features[i].replace("winner", "blue")
	features[i] = features[i].replace("loser", "red")

data.columns = features

data.to_csv('og.csv', index=False)
randomized_data.to_csv('swapped.csv', index=False)