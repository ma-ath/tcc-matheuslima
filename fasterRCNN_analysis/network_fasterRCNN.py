"""
    This a simple script to test somethings with the fasterRCNN network
"""
import json
from tqdm import tqdm
import numpy as np

MIN_SCORE = 0.5
FEATURES_PER_FRAME = 10

class_labels = {
    "car" : 1,
    "tv" : 2,
    "traffic light" : 3,
    "bicycle" : 4,
    "person" : 5,
    "bottle" : 6,
    "potted plant" : 7,
    "motorcycle" : 8,
    "boat" : 9,
    "truck" : 10,
    "sports ball" : 11,
    "horse" : 12,
    "tennis racket" : 13,
    "laptop" : 14,
    "bench" : 15,
    "bird" : 16,
    "skis" : 17,
    "bus" : 18,
    "oven" : 19,
    "suitcase" : 20,
    "microwave" : 21,
    "clock" : 22,
    "cell phone" : 23,
    "sink" : 24,
    "fire hydrant" : 25,
    "dog" : 26,
    "chair" : 27,
    "vase" : 28,
    "parking meter" : 29,
    "train" : 30,
    "skateboard" : 31,
    "backpack" : 32,
    "snowboard" : 33,
    "handbag" : 34,
    "umbrella" : 35,
    "donut" : 36,
    "book" : 37,
    "baseball glove" : 38,
    "broccoli" : 39,
    "cow" : 40,
    "refrigerator" : 41,
    "cat" : 42,
    "wine glass" : 43,
    "cup" : 44,
    "airplane" : 45,
    "toaster" : 46,
    "baseball bat" : 47,
    "stop sign" : 48,
    "bowl" : 49,
    "zebra" : 50,
    "toothbrush" : 51,
    "dining table" : 52,
    "cake" : 53,
    "giraffe" : 54,
    "remote" : 55,
    "surfboard" : 56,
    "hair drier" : 57,
    "fork" : 58,
    "teddy bear" : 59,
    "hot dog" : 60,
    "mouse" : 61,
    "frisbee" : 62,
    "elephant" : 63
}

def JSON_to_tensor(json_name="M2U00001.json", MIN_SCORE=MIN_SCORE, FEATURES_PER_FRAME=FEATURES_PER_FRAME, class_labels=class_labels):
    with open("dataset/fasterRCNN_features/"+json_name, "r") as fp:
        features = json.load(fp)

    #   --------- Remoção de baixos score ---------
    #   Remove all features with score less than MIN_SCORE
    print("JSON tem tamanho:", len(features))
    print("Extraindo todos as features com score menor que",MIN_SCORE)
    i = 0

    pbar = tqdm(total=len(features))

    while i < len(features):
        if features[i]['score'] < MIN_SCORE:
            del features[i]
            i -= 1
        i += 1
        pbar.update()

    pbar.close()
    print("Tamanho do JSON após remoção:", len(features))

    """
    #   --------- Substituição de Labels ---------
    #   Substitui "category" por um número
    print("Substituindo labels", MIN_SCORE)
    i = 0
    pbar = tqdm(total=len(features))
    while i < len(features):
        try:
            features[i]['category'] = class_labels[features[i]['category']]
            i += 1
            pbar.update()
        except:
            print("Feature not in class map:",features[i]['category'])
            pbar.close()        
            exit()
    pbar.close()
    print("All labels were substituted")
    """
    #   Cria e preenche o vetor numpy de saida.
    #   o vetor de saída é da seguinte forma: [Frame,Caracteristicas,categoria,score,w1,w2,h1,h2]
    #   Número de frames é o ultimo frame da ultima feature
    number_of_frames = features[len(features)-1]['frame']
    print("Número total de frames:", number_of_frames)
    number_of_categories = len(class_labels)
    print("Número total de categorias:", number_of_categories)

    output_tensor = np.zeros(
                        shape=(number_of_frames, FEATURES_PER_FRAME,number_of_categories, 5)
                        )

    print("Output tensor shape:", output_tensor.shape)

    def category_to_vector(category,labels=class_labels):
        vector = np.zeros(shape=(len(labels)))
        vector[labels[category]-1] = 1
        return vector

    pbar = tqdm(total=len(features))

    frame_counter = 0
    feature_counter = 0
    index = 0
    while frame_counter < number_of_frames:
        if features[index]['frame'] == frame_counter:
            if feature_counter < FEATURES_PER_FRAME:
                one_hot = category_to_vector(features[index]['category'])

                feature =  np.array([features[index]['score'],
                            features[index]['bbox'][0],
                            features[index]['bbox'][1],
                            features[index]['bbox'][2],
                            features[index]['bbox'][3]])
                
                feature_concat = np.zeros(shape=(number_of_categories, 5))
                for i in range(number_of_categories):
                    if one_hot[i] == 1:
                        feature_concat[i] = feature
                
                output_tensor[frame_counter][feature_counter] = feature_concat

                feature_counter += 1
            index += 1
            pbar.update()
        else:
            frame_counter += 1
            feature_counter = 0
    pbar.close()

    return output_tensor

if __name__ == '__main__':
    js = 'M2U00006.json'
    tensor = JSON_to_tensor(js)

    print("Salvando tensor no disco")
    np.save("fasterRCNN_analysis/"+js, tensor)