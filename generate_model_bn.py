import cv2
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

def split_data(dataset, ratio=0.85):
    index = int(len(dataset) * ratio)
    return dataset[:index], dataset[index:]

# convert img to pixel
images = []
list_file_dogs = os.listdir('data/train/dogs')
list_file_gardens = os.listdir('data/train/gardens')

# fill dogs photos
for filename in tqdm(list_file_dogs[]):
    path_img = os.path.join('data/train/dogs', filename)
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    images.append([np.array(img_pixel), np.array([1, 0])])

# fill gardens photos
for filename in tqdm(list_file_gardens[]):
    path_img = os.path.join('data/train/gardens', filename)
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    images.append([np.array(img_pixel), np.array([0, 1])])

random.shuffle(images)
train, test = split_data(images)

train_data = np.array([i[0] for i in train]).reshape(-1, 64, 64, 3)
train_label = np.array([i[1] for i in train])
test_data = np.array([i[0] for i in test]).reshape(-1, 64, 64, 3)
test_label = np.array([i[1] for i in test])

# CREATE MODEL

model = Sequential()

model.add(InputLayer(input_shape=[64, 64, 3]))

model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=1e-4)

# compile and fit the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_data, y=train_label, epochs=50, batch_size=128, validation_split=0.1)

# evaluate the model
scores = model.evaluate(train_data, train_label, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# MODEL TO JSON

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
