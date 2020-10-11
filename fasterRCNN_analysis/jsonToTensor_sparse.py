"""
    This a simple script to test somethings with the fasterRCNN network
"""
import json
from tqdm import tqdm
import numpy as np
import math as m

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

    #   Cria e preenche o vetor numpy de saida.
    #   o vetor de saída é da seguinte forma: [Frame,Caracteristicas,categoria,score,w1,w2,h1,h2]
    #   Número de frames é o ultimo frame da ultima feature

    output_tensor = [None]*len(features)
    cont = 0
    for feature in features:
        number_of_frames = feature[len(feature)-1]['frame']
        print("Número total de frames:", number_of_frames)
        number_of_categories = len(category_list)
        print("Número total de categorias:", number_of_categories)

        output_tensor[cont] = np.zeros(
                            shape=(number_of_frames, FEATURES_PER_FRAME, number_of_categories, 2)
                            )

        print("Output tensor shape:", output_tensor[cont].shape)

        pbar = tqdm(total=len(feature))

        frame_counter = 0
        feature_counter = 0
        index = 0
        while frame_counter < number_of_frames:
            if feature[index]['frame'] == frame_counter:
                if feature_counter < FEATURES_PER_FRAME:
                    one_hot = category_to_vector(feature[index]['category'], category_list)
                    
                    #feat =  np.array([feature[index]['score'],
                    #            feature[index]['bbox'][0],
                    #            feature[index]['bbox'][1],
                    #            feature[index]['bbox'][2],
                    #            feature[index]['bbox'][3]])
                    
                    size = (feature[index]['bbox'][0]-feature[index]['bbox'][1])*(feature[index]['bbox'][2]-feature[index]['bbox'][3])
                    size = m.sqrt(abs(size))
                    feat =  np.array([feature[index]['score'],size])

                    feature_concat = np.zeros(shape=(number_of_categories, len(feat)))
                    for i in range(number_of_categories):
                        if one_hot[i] == 1:
                            feature_concat[i] = feat
                    
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
        'M2U00014.json']

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
        'cow'
    ]

    tensors = JSON_to_tensor(json_list, FEATURES_PER_FRAME=5, MIN_SCORE=0.6, BLOCK_LIST=block_list)
    for i in range(len(tensors)):
        print("Salvando tensor no disco")
        np.save("fasterRCNN_analysis/"+json_list[i]+".sparse", tensors[i])
        sparsity = 1.0 - count_nonzero(tensors[i]) / tensors[i].size
        print(json_list[i], "sparsity:", sparsity)