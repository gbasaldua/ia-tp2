import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json

# Parametros del modelo

length, height = 150, 150
lr = 0.0005
model_path = './model/model.json'
weights_model_path = './model/weights.h5'

# Path de imagenes de prueba
path_dogs = './data/test/dogs'
path_gardens = './data/test/gardens'

print("Carga del modelo de prediccion")

# se obtiene el modelos en formato json
json_file = open(model_path, 'r')
loaded_model = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model)

# se obtienen los pesos
cnn.load_weights(weights_model_path)

# compilacion del modelo con:
#  funcion de perdida categorical_crossentropy
#  funcion Adam con learning rate = 0.0005
#  metricas accuracy
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


# SHOW RESULT
list_file_dogs = os.listdir(path_dogs)
list_file_gardens = os.listdir(path_gardens)

# fig = plt.figure(figsize=(14, 14))

print("Perros => ")

# se obtienen las imagenes de perros y se muestran las que son detectados como jardines
for index, filename in enumerate(list_file_dogs):

    # se forma el path completo de la imagen
    path_img = os.path.join(path_dogs, filename)

    # se obtiene la imagen
    img_color = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_color, (height, length)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, height, length, 3)
    model_out = cnn.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        str_label = 'pred: Perro'
    else:
        str_label = 'pred: Jadin'

    if str_label == 'pred: Jadin':
        print(path_img)
        # y = fig.add_subplot(6, 5, index+1)
        
        # y.imshow(img_color)
        # plt.title(str_label)
        # y.axes.get_xaxis().set_visible(False)
        # y.axes.get_yaxis().set_visible(False)


print("Jardines => ")

# se obtienen las imagenes de jardines y se muestran las que son detectados como perros
for index, filename in enumerate(list_file_gardens):

    # se forma el path completo de la imagen
    path_img = os.path.join(path_gardens, filename)

    # print(filename)
    # se obtiene la imagen
    img_color = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_color, (height, length)) #150 x 150 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, height, length, 3)
    model_out = cnn.predict([data])

    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        str_label = 'pred: Perro'
    else:
        str_label = 'pred: Jardin'

    if str_label == 'pred: Perro':
        print(path_img)
        # y = fig.add_subplot(6, 5, index+6)

        # y.imshow(img_color)
        # plt.title(str_label)
        # y.axes.get_xaxis().set_visible(False)
        # y.axes.get_yaxis().set_visible(False)

# plt.show()
