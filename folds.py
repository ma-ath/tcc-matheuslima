"""
    This file contains a
    Python dictionaries with all training and testing folds
    Each entry has the name of training and testing videos
"""
folds_number = 3
folds = [dict() for i in range(folds_number)]

for i in range(folds_number):
    folds[i]["name"] = "fold_"+str(i)
    folds[i]["number"] = i

folds[0]["training_videos"] = [
    "twice_yesoryes.mpg",
    "twice_dancethenight.mpg"
]
folds[0]["testing_videos"] = [
    "twice_fancy.mpg"
]

folds[1]["training_videos"] = [
    "twice_yesoryes.mpg",
    "twice_fancy.mpg"
]
folds[1]["testing_videos"] = [
    "twice_dancethenight.mpg"
]

folds[2]["training_videos"] = [
    "twice_dancethenight.mpg"
]
folds[2]["testing_videos"] = [
    "twice_yesoryes.mpg",
    "twice_fancy.mpg"
]