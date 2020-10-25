"""
    This file contains a
    Python dictionaries with all training and testing folds
    Each entry has the name of training and testing videos
"""
folds_number = 10
folds = [dict() for i in range(folds_number)]

for i in range(folds_number):
    folds[i]["name"] = "fold_"+str(i)
    folds[i]["number"] = i

folds[0]["training_videos"] = [
    "M2U00004.MPG",
    "M2U00006.MPG",
    "M2U00008.MPG",
    "M2U00012.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00016.MPG",
    "M2U00018.MPG",
    "M2U00022.MPG",
    "M2U00024.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00035.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[0]["testing_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00005.MPG",
    "M2U00007.MPG",
    "M2U00019.MPG",
    "M2U00023.MPG",
    "M2U00025.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00036.MPG"
]

folds[1]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00008.MPG",
    "M2U00012.MPG",
    "M2U00014.MPG",
    "M2U00015.MPG",
    "M2U00018.MPG",
    "M2U00019.MPG",
    "M2U00022.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00025.MPG",
    "M2U00029.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[1]["testing_videos"] = [
    "M2U00004.MPG",
    "M2U00007.MPG",
    "M2U00017.MPG",
    "M2U00016.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00030.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG"
]

folds[2]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00007.MPG",
    "M2U00008.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00016.MPG",
    "M2U00018.MPG",
    "M2U00019.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00025.MPG",
    "M2U00027.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]

folds[2]["testing_videos"] = [
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00012.MPG",
    "M2U00015.MPG",
    "M2U00022.MPG",
    "M2U00026.MPG",
    "M2U00035.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG"
]

folds[3]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00008.MPG",
    "M2U00012.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00016.MPG",
    "M2U00019.MPG",
    "M2U00022.MPG",
    "M2U00023.MPG",
    "M2U00025.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00030.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG",
    "M2U00036.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[3]["testing_videos"] = [
    "M2U00014.MPG",
    "M2U00018.MPG",
    "M2U00024.MPG",
    "M2U00029.MPG",
    "M2U00031.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG"
]

folds[4]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00016.MPG",
    "M2U00019.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00025.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG"
]
folds[4]["testing_videos"] = [
    "M2U00003.MPG",
    "M2U00008.MPG",
    "M2U00012.MPG",
    "M2U00018.MPG",
    "M2U00022.MPG",
    "M2U00036.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[5]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00004.MPG",
    "M2U00012.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00016.MPG",
    "M2U00022.MPG",
    "M2U00025.MPG",
    "M2U00026.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[5]["testing_videos"] = [
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00008.MPG",
    "M2U00018.MPG",
    "M2U00019.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00027.MPG"
]

folds[6]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00008.MPG",
    "M2U00016.MPG",
    "M2U00018.MPG",
    "M2U00019.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00025.MPG",
    "M2U00027.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[6]["testing_videos"] = [
    "M2U00012.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00022.MPG",
    "M2U00026.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG"
]

folds[7]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00008.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00016.MPG",
    "M2U00018.MPG",
    "M2U00022.MPG",
    "M2U00023.MPG",
    "M2U00025.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00050.MPG"
]
folds[7]["testing_videos"] = [
    "M2U00003.MPG",
    "M2U00012.MPG",
    "M2U00015.MPG",
    "M2U00019.MPG",
    "M2U00024.MPG",
    "M2U00029.MPG",
    "M2U00032.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG"
]

folds[8]["training_videos"] = [
    "M2U00003.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00012.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00016.MPG",
    "M2U00019.MPG",
    "M2U00022.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00042.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00047.MPG",
    "M2U00050.MPG"
]
folds[8]["testing_videos"] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00004.MPG",
    "M2U00008.MPG",
    "M2U00018.MPG",
    "M2U00025.MPG",
    "M2U00035.MPG",
    "M2U00039.MPG",
    "M2U00041.MPG",
    "M2U00046.MPG",
    "M2U00048.MPG"
]

folds[9]["training_videos"] = [
    "M2U00001.MPG",
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00008.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00018.MPG",
    "M2U00022.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00025.MPG",
    "M2U00026.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00035.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00041.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
folds[9]["testing_videos"] = [
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00007.MPG",
    "M2U00012.MPG",
    "M2U00016.MPG",
    "M2U00019.MPG",
    "M2U00027.MPG",
    "M2U00033.MPG",
    "M2U00042.MPG",
    "M2U00047.MPG"
]



"""
folds[...]["..."] = [
    "M2U00001.MPG",
    "M2U00002.MPG",
    "M2U00003.MPG",
    "M2U00004.MPG",
    "M2U00005.MPG",
    "M2U00006.MPG",
    "M2U00007.MPG",
    "M2U00008.MPG",
    "M2U00012.MPG",
    "M2U00014.MPG",
    "M2U00017.MPG",
    "M2U00015.MPG",
    "M2U00016.MPG",
    "M2U00018.MPG",
    "M2U00019.MPG",
    "M2U00022.MPG",
    "M2U00023.MPG",
    "M2U00024.MPG",
    "M2U00025.MPG",
    "M2U00026.MPG",
    "M2U00027.MPG",
    "M2U00029.MPG",
    "M2U00030.MPG",
    "M2U00031.MPG",
    "M2U00032.MPG",
    "M2U00033.MPG",
    "M2U00035.MPG",
    "M2U00036.MPG",
    "M2U00037.MPG",
    "M2U00039.MPG",
    "M2U00041.MPG",
    "M2U00042.MPG",
    "M2U00043.MPG",
    "M2U00045.MPG",
    "M2U00046.MPG",
    "M2U00047.MPG",
    "M2U00048.MPG",
    "M2U00050.MPG"
]
"""

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