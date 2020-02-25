# Código de: https://www.tensorflow.org/tutorials/keras/classification

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#
#   IMPORTAÇÃO E VISUALIZAÇÃO DOS DADOS
#

# Faz o download do dataset fashion mnist
fashion_mnist = keras.datasets.fashion_mnist

# Separa o dataset em dois grupos, grupo de treinamento e grupo de teste
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Array com o nome de cada um dos grupos de roupas
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Vamos fazer os dados do dataset irem de 0 a 1 apenas, dividindo tudo por 255

train_images = train_images / 255.0
test_images = test_images / 255.0

# Agora vamos mostrar as 25 primeiras imagens do dataset, com suas respectivas classes
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#
#   GERAÇÃO DO MODELO DE TREINAMENTO
#

#   Gera o modelo
#   The first layer in this network, tf.keras.layers.Flatten,
#   transforms the format of the images from a two-dimensional
#   array (of 28 by 28 pixels) to a one-dimensional array
#   (of 28 * 28 = 784 pixels). Think of this layer as unstacking
#   rows of pixels in the image and lining them up. This layer has
#   no parameters to learn; it only reformats the data.
#
#   After the pixels are flattened, the network consists of a sequence
#   of two tf.keras.layers.Dense layers. These are densely connected,
#   or fully connected, neural layers. The first Dense layer has 128 nodes
#   (or neurons). The second (and last) layer is a 10-node softmax layer
#   that returns an array of 10 probability scores that sum to 1. Each node
#   contains a score that indicates the probability that the current image
#   belongs to one of the 10 classes.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

#   Before the model is ready for training, it needs a few more settings.
#   These are added during the model's compile step:
#   - Loss function —This measures how accurate the model is during training.
#   You want to minimize this function to "steer" the model in the right direction.
#   - Optimizer —This is how the model is updated based on the data it sees and its loss function.
#   - Metrics —Used to monitor the training and testing steps. The following example
#   uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#
#   TREINAMENTO DA REDE
#

#   O treinamento é feito com base no dados de TREINAMENTO
model.fit(train_images, train_labels, epochs=10)

#
#   TESTE DO MODELO
#

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)

#
#   USANDO O MODELO TREINADO PARA FAZER PREDIÇÕES
#

#   Coloco na camada de saída uma função de ativação softmax(),
#   para poder interpretar os resultados como probabilidades
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#   E faço as predições
predictions = probability_model.predict(test_images)

#   Uma predição é, nesse caso, um array de 10 valores com as probabilidades
#   da imagem ser tal artigo de roupa. Para saber qual artigo a rede acha que
#   o artigo x é, basta pegar a maior probabilidade.
#   Por exemplo, para a primeira imagem:
print(np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()