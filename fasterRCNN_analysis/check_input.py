if __name__ == '__main__':
    import numpy as np
    import pickle
    import json

    json_list = [
        'M2U00001.json',
        'M2U00002.json',
        'M2U00003.json',
        'M2U00004.json',
        'M2U00005.json',
        'M2U00006.json',
        'M2U00007.json',
        'M2U00008.json',
        'M2U00012.json',
        'M2U00014.json',
        "M2U00015.json",
        "M2U00016.json",
        "M2U00017.json",
        "M2U00018.json",
        "M2U00019.json",
        "M2U00022.json",
        "M2U00023.json",
        "M2U00024.json",
        "M2U00025.json",
        "M2U00026.json",
        "M2U00027.json",
        "M2U00029.json",
        "M2U00030.json",
        "M2U00031.json",
        "M2U00032.json",
        "M2U00033.json",
        "M2U00035.json",
        "M2U00036.json",
        "M2U00037.json",
        "M2U00039.json",
        "M2U00041.json",
        "M2U00042.json",
        "M2U00043.json",
        "M2U00045.json",
        "M2U00046.json",
        "M2U00047.json",
        "M2U00048.json",
        "M2U00050.json"
    ]

    video_number = [1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    12,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    29,
                    30,
                    31,
                    32,
                    33,
                    35,
                    36,
                    37,
                    39,
                    41,
                    42,
                    43,
                    45,
                    46,
                    47,
                    48,
                    50
    ]

    #   Carrega lista de Json
    i = 0
    myJson = [None]*len(json_list)
    for json_name in json_list:
        with open("dataset/fasterRCNN_features/"+json_name, "r") as fp:
            myJson[i] = json.load(fp)
            i += 1
    
    for i in range(len(json_list)):
        print(json_list[i])

        with open("fasterRCNN_analysis/nmb_of_frames_"+str(video_number[i]), "rb") as fp:
            nmb_of_frames = pickle.load(fp)
        
        frame_now = 0
        for j in range(len(myJson[i])):
            if int(myJson[i][j]['frame']) != frame_now:
                if int(myJson[i][j]['frame']) == frame_now+1:
                    frame_now += 1
                else:
                    print("frame_now", frame_now, "myJson[i][j]['frame']", myJson[i][j]['frame'])
                    frame_now = int(myJson[i][j]['frame'])
        print("Ultimo frame:", myJson[i][len(myJson[i])-1]['frame'])
        print("nmb_of_frames:", nmb_of_frames-1)

    for i in range(len(json_list)):
        myArray = np.load("dataset/fasterRCNN_features/"+json_list[i]+".sparse.npy")

        with open("fasterRCNN_analysis/nmb_of_frames_"+str(video_number[i]), "rb") as fp:
            nmb_of_frames = pickle.load(fp)

        if myArray.shape[0] != nmb_of_frames:
            print(json_list[i], "shape:", myArray.shape, "nmb_of_frames:", nmb_of_frames)