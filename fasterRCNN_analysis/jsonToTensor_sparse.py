"""
    This a simple script to test somethings with the fasterRCNN network
"""
import json
from tqdm import tqdm
import numpy as np
import math as m
import pickle

def JSON_to_tensor(json_list, MIN_SCORE=0.5, FEATURES_PER_FRAME=10, BLOCK_LIST = []):

    def category_to_vector(category, labels):
        vector = np.zeros(shape=(len(labels)))
        place = [i for i, x in enumerate(labels) if x == category]
        vector[place[0]] = 1
        return vector

    print("Loading JSONs from json_list...")
    features = [None]*len(json_list)
    i = 0
    for json_name in json_list:
        with open("dataset/fasterRCNN_features/"+json_name, "r") as fp:
            features[i] = json.load(fp)
            i += 1

    #   --------- Remoção de baixos score ---------
    #   Remove all features with score less than MIN_SCORE
    j = 0
    for feature in features:
        print("Processing JSON", json_list[j])
        print("JSON has size:", len(feature))
        print("Removing features with score less than", MIN_SCORE)
        i = 0
        pbar = tqdm(total=len(feature))
        while i < len(feature):
            if feature[i]['score'] < MIN_SCORE or feature[i]['category'] in BLOCK_LIST:
                del feature[i]
                i -= 1
            i += 1
            pbar.update()
        pbar.close()
        print("JSON has now size:", len(feature))
        j += 1
    #   --------- Label generation ---------

    category_list = []

    print("Looking for all categories")
    for feature in features:
        i = 0
        while i < len(feature):
            unknown_category = feature[i]['category'] not in category_list
            if unknown_category:
                print("Found an unknown category:", feature[i]['category'])
                category_list.append(feature[i]['category'])
            i += 1
    print("All categories were processed")
    for category in category_list:
        print(category)

    #   --------- Checar   ---------
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

    for i in range(len(json_list)):
        print(json_list[i])

        with open("fasterRCNN_analysis/nmb_of_frames_"+str(video_number[i]), "rb") as fp:
            nmb_of_frames = pickle.load(fp)
        
        frame_now = 0
        for j in range(len(features[i])):
            if int(features[i][j]['frame']) != frame_now:
                if int(features[i][j]['frame']) == frame_now+1:
                    frame_now += 1
                else:
                    # print("frame_now", frame_now, "features[i][j]['frame']", features[i][j]['frame'])
                    frame_now = int(features[i][j]['frame'])
        print("Ultimo frame:", features[i][len(features[i])-1]['frame'])
        print("nmb_of_frames:", nmb_of_frames-1)

    #   Cria e preenche o vetor numpy de saida.
    #   o vetor de saída é da seguinte forma: [Frame,nº de caracteristicas,caracteristicas(onehot+score+size)]
    #   Número de frames é o ultimo frame da ultima feature

    output_tensor = [None]*len(features)
    cont = 0
    for feature in features:
        with open("fasterRCNN_analysis/nmb_of_frames_"+str(video_number[cont]), "rb") as fp:
            number_of_frames = pickle.load(fp)
        # number_of_frames = feature[len(feature)-1]['frame']+1
        print("Número total de frames:", number_of_frames)
        number_of_categories = len(category_list)
        print("Número total de categorias:", number_of_categories)

        output_tensor[cont] = np.zeros(
                            shape=(number_of_frames, FEATURES_PER_FRAME, number_of_categories+2)
                            )

        print("Output tensor shape:", output_tensor[cont].shape)

        pbar = tqdm(total=len(feature))

        frame_counter = 0
        feature_counter = 0
        index = 0
        while frame_counter < number_of_frames and index < len(feature):
            if feature[index]['frame'] == frame_counter:
                if feature_counter < FEATURES_PER_FRAME:
                    one_hot = category_to_vector(feature[index]['category'], category_list)
          
                    size = (feature[index]['bbox'][0]-feature[index]['bbox'][1])*(feature[index]['bbox'][2]-feature[index]['bbox'][3])
                    size = m.sqrt(abs(size))
                    feat =  np.array([feature[index]['score'], size])

                    feature_concat = np.concatenate((feat, one_hot))
                  
                    output_tensor[cont][frame_counter][feature_counter] = feature_concat

                    feature_counter += 1
                index += 1
                pbar.update()
            else:
                frame_counter += 1
                feature_counter = 0
        pbar.close()
        cont += 1

    return output_tensor

if __name__ == '__main__':
    from numpy import count_nonzero

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

    block_list = [
        'suitcase',
        'boat',
        'tennis racket',
        'tv',
        'baseball glove',
        'backpack',
        'chair',
        'bird',
        'dog',
        'bench',
        'cell phone',
        'skateboard',
        'potted plant',
        'vase',
        'sports ball',
        'bottle',
        'cup',
        'wine glass',
        'frisbee',
        'skis',
        'sink',
        'remote',
        'umbrella',
        'fire hydrant',
        'train',
        'broccoli',
        'donut',
        'teddy bear',
        'airplane',
        'bowl',
        'book',
        'cat',
        'handbag',
        'zebra',
        'baseball bat',
        'snowboard',
        'banana',
        'horse',
        'stop sign',
        'clock',
        'elephant',
        'cow',
        'parking meter',
        'traffic light',
        'toilet',
        'fork',
        'kite',
        'giraffe',
        'tie'
    ]

    tensors = JSON_to_tensor(json_list, FEATURES_PER_FRAME=20, MIN_SCORE=0.7, BLOCK_LIST=block_list)
    for i in range(len(tensors)):
        print("Salvando tensor no disco")
        np.save("dataset/fasterRCNN_features/"+json_list[i]+".sparse", tensors[i])
        sparsity = 1.0 - count_nonzero(tensors[i]) / tensors[i].size
        print(json_list[i], "sparsity:", sparsity)