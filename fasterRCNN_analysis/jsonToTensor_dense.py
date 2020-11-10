"""
    This a simple script to test somethings with the fasterRCNN network
"""
import json
from tqdm import tqdm
import numpy as np
import math as m

def JSON_to_tensor(json_list, MIN_SCORE=0.5, BLOCK_LIST = []):

    def category_counter_to_vector(category_counter):
        vector = [0]*len(category_counter)

        i = 0
        for key, value in category_counter.items():
            vector[i] = value
            i += 1

        vector = np.array(vector)
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

    category_counter = {}

    for category in category_list:
        category_counter[category] = 0

    #   --------- Counting ---------
    for feature in features:
        i = 0
        while i < len(feature):
            category_counter[feature[i]['category']] += 1
            i += 1

    #   Order by counter
    category_counter = {k: v for k, v in sorted(category_counter.items(), key=lambda item: item[1], reverse=True)}

    print("Total number of apearences per category:")
    for key, value in category_counter.items():
        print(key, ":", value)

    #   Cria e preenche o vetor numpy de saida.
    #   o vetor de saída é da seguinte forma: [Frame,categoria,numero_de_repeticoes]
    #   Número de frames é o ultimo frame da ultima feature

    output_tensor = [None]*len(features)
    cont = 0
    for feature in features:
        number_of_frames = feature[len(feature)-1]['frame'] +1
        print("Total number of frames:", number_of_frames)
        number_of_categories = len(category_list)
        print("Total number of categories:", number_of_categories)

        output_tensor[cont] = np.zeros(
                            shape=(number_of_frames, number_of_categories)
                            )

        print("Output tensor shape:", output_tensor[cont].shape)

        pbar = tqdm(total=len(feature))

        frame_counter = 0
        index = 0
        category_counter = {}

        for category in category_list:
            category_counter[category] = 0

        while frame_counter <= number_of_frames and index < len(feature):
            if feature[index]['frame'] == frame_counter:
                #   Count number of apearences of this categorie
                #   --------- Counting ---------
                category_counter[feature[index]['category']] += 1
                index += 1
                pbar.update()
            else:
                feat = category_counter_to_vector(category_counter)
                
                output_tensor[cont][frame_counter] = feat

                frame_counter += 1              #   Next frame
                for category in category_list:  #   category_counter = 0
                    category_counter[category] = 0
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

    tensors = JSON_to_tensor(json_list, MIN_SCORE=0.70, BLOCK_LIST=block_list)
    for i in range(len(tensors)):
        print("Salvando tensor no disco")
        np.save("dataset/fasterRCNN_features/"+json_list[i]+".dense", tensors[i])
        sparsity = 1.0 - count_nonzero(tensors[i]) / tensors[i].size
        print(json_list[i], "sparsity:", sparsity)