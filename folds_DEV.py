"""
    This file contains a
    Python dictionaries with all training and testing folds
    Each entry has the name of training and testing videos
"""
folds_number = 1
folds = [dict() for i in range(folds_number)]

for i in range(folds_number):
    folds[i]["name"] = "fold_"+str(i)
    folds[i]["number"] = i

folds[0]["training_videos"] = [
    "M2U00003.MPG"
]
folds[0]["testing_videos"] = [
    "M2U00004.MPG"
]

if __name__ == "__main__":
    print("The following new videos were added to the dataset")
    for fold in folds:
        print(fold['name'])
        print('training_videos')
        for video in fold['training_videos']:
            print('\t'+video)
        print('testing_videos')
        for video in fold['testing_videos']:
            print('\t'+video)