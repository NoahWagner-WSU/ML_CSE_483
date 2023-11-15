import pandas as pd
import numpy as np

data = pd.read_csv("BattlesStaging_01012021_WL_tagged.csv", nrows=100000)

remove_columns = [
	"battleTime",
	"arena.id",
	"gameMode.id",
	"average.startingTrophies",
	"winner.tag",
	"winner.startingTrophies",
	"winner.trophyChange",
	"winner.crowns",
	"winner.clan.tag",
	"winner.clan.badgeId",
	"loser.tag",
	"loser.startingTrophies",
	"loser.trophyChange",
	"loser.crowns",
	"loser.clan.tag",
	"loser.clan.badgeId",
	"tournamentTag",
	"winner.troop.count",
	"winner.structure.count",
	"winner.spell.count",
	"winner.common.count",
	"winner.rare.count",
	"winner.epic.count",
	"winner.legendary.count",
	"winner.cards.list",
	"loser.cards.list",
	"loser.troop.count",
	"loser.structure.count",
	"loser.spell.count",
	"loser.common.count",
	"loser.rare.count",
	"loser.epic.count",
	"loser.legendary.count",
]

cleaned = data.drop(columns=remove_columns)

print(cleaned)