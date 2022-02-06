
import os
import cv2
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout#model evaluation packages
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix,classification_report


train_data_path = r"C:\UpworkProjects\DwayneSalo\skindisease\train"
validation_data_path =  r"C:\UpworkProjects\DwayneSalo\skindisease\test"


CLASSES, gems = [], [] 

for root, dirs, files in os.walk(r"C:\UpworkProjects\DwayneSalo\skindisease"):
    f = os.path.basename(root)    # get class name - Amethyst, Onyx, etc    
        
    if len(files) > 0:
        gems.append(len(files))
        if f not in CLASSES:
            CLASSES.append(f) 
    

gems_count = len(CLASSES) 
print('{} classes with {} images in total'.format(len(CLASSES), sum(gems)))


img_w, img_h = 220, 220    
train_dir = r"C:\UpworkProjects\DwayneSalo\skindisease\train"



def read_imgs_lbls(_dir):
    Images, Labels = [], []
    for root, dirs, files in os.walk(_dir):
        f = os.path.basename(root)       
        for file in files:
            Labels.append(f)
            try:
                image = cv2.imread(root+'/'+file)             
                image = cv2.resize(image,(int(img_w*1.5), int(img_h*1.5)))       
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Images.append(image)
            except Exception as e:
                print(e)
    Images = np.array(Images)
    return (Images, Labels)


def get_class_index(Labels):
    for i, n in enumerate(Labels):
        for j, k in enumerate(CLASSES):    # foreach CLASSES
            if n == k:
                Labels[i] = j
    Labels = np.array(Labels)
    return Labels


#Train_Imgs, Train_Lbls = read_imgs_lbls(train_dir)
#Train_Lbls = get_class_index(Train_Lbls)
#print('Shape of train images: {}'.format(Train_Imgs.shape))
#print('Shape of train labels: {}'.format(Train_Lbls.shape))




#dim = 4 

#f,ax = plt.subplots(dim,dim) 
#f.subplots_adjust(0,0,2,2)
#for i in range(0,dim):
 #  for j in range(0,dim):
  #      rnd_number = randint(0,len(Train_Imgs))
   #     cl = Train_Lbls[rnd_number]
    #    ax[i,j].imshow(Train_Imgs[rnd_number])
     #   ax[i,j].set_title(CLASSES[cl]+': ' + str(cl))
      #  ax[i,j].axis('off')


training_data_generator = ImageDataGenerator(rescale=1./255,
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                             horizontal_flip=True,
                                            fill_mode="nearest",)


training_datas = training_data_generator.flow_from_directory(train_data_path,
                                            target_size=(150, 150),
                                           batch_size=32,
                                           class_mode='binary')



training_datas.class_indices


my_dict = training_datas.class_indices




my_dict.keys()


categories = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']





DATADIR = train_data_path

CATEGORIES = categories

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='binary')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!




print(img_array)


IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()



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



validation_data_generator = ImageDataGenerator(rescale=1./255)



validation_data = validation_data_generator.flow_from_directory(validation_data_path,
                                            target_size=(150, 150),
                                           batch_size=32,
                                           class_mode='binary')



def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
 
images = [training_data[1][0][0] for i in range(5)]
plotImages(images)


#Building cnn model
cnn_model = keras.models.Sequential([
                                    keras.layers.Conv2D(filters=32, kernel_size=7, input_shape=[150, 150, 3]),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                      
                                    keras.layers.Conv2D(filters=64, kernel_size=5),
                              
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                      
                                    keras.layers.Conv2D(filters=128, kernel_size=3),
                                  
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                                                          
                                    keras.layers.Conv2D(filters=256, kernel_size=3),
                            
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                                                      
                                    keras.layers.Flatten(), # neural network beulding
                                    keras.layers.Dense(units=128, activation='relu'), # input layers
                                 
                                    keras.layers.Dropout(0.5),                                      
                                    keras.layers.Dense(units=256, activation='relu'),  
                                                             
                                    keras.layers.Dropout(0.5),                                    
                                    keras.layers.Dense(units=23, activation='softmax') # output layer
])




from tensorflow.keras.optimizers import Adam
cnn_model.compile(optimizer = Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])




from keras.callbacks import ModelCheckpoint
model_path = r'C:\UpworkProjects\DwayneSalo\skindisease\skin_disease.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



history = cnn_model.fit(training_datas, 
                          epochs=100, 
                          verbose=1, 
                            callbacks=callbacks_list)

history = cnn_model.fit(training_datas, 
                          epochs=100, 
                          verbose=1, 
                           validation_data= validation_data,
                            callbacks=callbacks_list)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.show()


plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Train_Imgs, Train_Lbls, shuffle = True, test_size = 0.2, random_state = 42)
print('Shape of X_train: {}, y_train: {} '.format(X_train.shape, y_train.shape))
print('Shape of X_val: {}, y_val: {} '.format(X_test.shape, y_test.shape))


#reshape data from 4-D to 2-D array
X_train = X_train.reshape(2367, 326700)
X_test = X_test.reshape(592, 326700)#feature scaling
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()#fit and transform training dataset
X_train = minmax.fit_transform(X_train)#transform testing dataset
X_test = minmax.transform(X_test)
print('Number of unique classes: ', len(np.unique(y_train)))
print('Classes: ', np.unique(y_train))
X_train.shape


classifier_e25 = Sequential()#add 1st hidden layer
classifier_e25.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))#add output layer
classifier_e25.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))#compile the neural network
classifier_e25.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])#model summary
classifier_e25.summary()

#fit training dataset into the model
classifier_e25_fit = classifier_e25.fit(X_train, y_train, epochs=25, verbose=0)

#evaluate the model for testing dataset
test_loss_e25 = classifier_e25.evaluate(X_test, y_test, verbose=0)#calculate evaluation parameters
f1_e25 = f1_score(y_test, classifier_e25.predict_classes(X_test), average='micro')
roc_e25 = roc_auc_score(y_test, classifier_e25.predict_proba(X_test), multi_class='ovo')#create evaluation dataframe
stats_e25 = pd.DataFrame({'Test accuracy' :  round(test_loss_e25[1]*100,3),
                      'F1 score'      : round(f1_e25,3),
                      'ROC AUC score' : round(roc_e25,3),
                      'Total Loss'    : round(test_loss_e25[0],3)}, index=[0])#print evaluation dataframe
confusion_matrix(y_test, np.argmax(classifier_e25.predict(X_test),axis=1))
confusion = classification_report(y_test, np.argmax(classifier_e25.predict(X_test),axis=1))
print(confusion)