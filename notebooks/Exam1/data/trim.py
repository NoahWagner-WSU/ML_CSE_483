import pandas as pd
import  sys

data = pd.read_csv("game_data_all.csv")

final = data.drop(data.index[10000:])

final = final[["game", "release", "peak_players", "positive_reviews", "negative_reviews", "total_reviews", "rating", "players_right_now"]]

final.to_csv(sys.stdout, sep=',', index=False)