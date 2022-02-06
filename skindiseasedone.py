
#Import Libraries

import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

# Load the Training and Testing Dataset

train_data_path = r"C:\UpworkProjects\DwayneSalo\skindisease\train_n"
validation_data_path =  r"C:\UpworkProjects\DwayneSalo\skindisease\test_n"
training_data_generator = ImageDataGenerator(rescale=1./255)
training_data = training_data_generator.flow_from_directory(train_data_path,
                                            target_size=(128, 128),
                                           batch_size=32,
                                           class_mode='binary')

my_dict = training_data.class_indices
categories = ['Acne and Rosacea Photos']
DATADIR = train_data_path

CATEGORIES = categories

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  
        plt.imshow(img_array, cmap='binary') 
        plt.show()  

        break  
    break  
# Print the Array of the Images
print(img_array)
#Print the Image Shape
print(img_array.shape)

IMG_SIZE = 100

#Plot a Grey Scale Image
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
new_array.shape

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
X_train = X

y_train = y

plt.imshow(X_train[1],cmap='gray')
plt.show()
new_array.shape

X_train = np.array(X_train)

y_train = np.array(y_train)

# Now considering the Testing Dataset

validation_data_generator = ImageDataGenerator(rescale=1./255)

validation_data = validation_data_generator.flow_from_directory(validation_data_path,
                                            target_size=(128, 128),
                                           batch_size=32,
                                           class_mode='binary')

my_dict = validation_data.class_indices

my_dict.keys()

categories = ['Acne and Rosacea Photos']

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = validation_data_path

CATEGORIES = categories

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='binary') 
        plt.show()  

        break  
    break  
# Considering the Image Size of 100
IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
new_array.shape

validation_data = []
def create_validation_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                validation_data.append([new_array, class_num])
            except Exception as e:
                pass
create_validation_data()
print(len(validation_data))

X = []
y = []

for features,label in validation_data:
    X.append(features)
    y.append(label)
    
    
X_test = X

y_test = y


plt.imshow(X_test[1],cmap='gray')
plt.show()
new_array.shape

# Checking the shape of test dataset and converting into array
X_test = np.array(X_test)

X_test.shape

y_test = np.array(y_test)

y_test.shape

X_train.shape

y_train.shape

X_train = X_train / 255.0

y_train = y_train / 255.0

# First we will be implementing Artificial Neural Network
from tensorflow.keras import layers, models


from keras.layers import Dropout

# We will be using Sequential Model
ann = models.Sequential([
        layers.Flatten(input_shape=(100,100,3)),
        layers.Dense(5000, activation='relu'),
        layers.Dense(3000, activation='relu'),
        layers.Dropout((0.5)),
        layers.Dense(2, activation='sigmoid')    
    ])

# We will be using SGD Optimizer
ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Dataset
ann.fit(X_train, y_train, epochs=1)
print(ann.evaluate(X_test,y_test))

# Testing the Dataset

y_pred = ann.predict(X_test)
print(y_pred)
y_pred = ann.predict(X_test)
print(y_pred)

# Now applying Convolutional Neural Network

cnn = models.Sequential([
    layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same',input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.5)),
    layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.5)),
    layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.5)),
    layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout((0.5)),
    layers.Dense(80, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])

# We are using Adam Optimizer and sparse_categorical_crossentropy Loss
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Dataset
cnn.fit(X_train, y_train, epochs=1)

# Testing the Dataset
print(cnn.evaluate(X_test,y_test))
y_pred = cnn.predict(X_test)
print(y_pred[:5])