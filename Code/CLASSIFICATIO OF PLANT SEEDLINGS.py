import numpy as np # MATRIX OPERATIONS
import pandas as pd # EFFICIENT DATA STRUCTURES
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import math # MATHEMATICAL OPERATIONS
import cv2 # IMAGE PROCESSING - OPENCV
from glob import glob # FILE OPERATIONS
import itertools

# KERAS AND SKLEARN MODULES
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# GLOBAL VARIABLES
scale = 70
seed = 7

path_to_images ='E:/DELHI(sonipat)(TRAINING)/CLASSIFICATION OF PLANT SEEDLING/plant-seedlings/INPUT IMAGES/*/*.jpg'
images = glob(path_to_images)
trainingset = []
traininglabels = []
num = len(images)
count = 1
#READING IMAGES AND RESIZING THEM
for i in images:
    print(str(count)+'/'+str(num),end='\r')
    trainingset.append(cv2.resize(cv2.imread(i),(scale,scale)))
    traininglabels.append(i.split('\\')[1])
    count=count+1
trainingset = np.asarray(trainingset)
traininglabels = pd.DataFrame(traininglabels)

new_train = []
sets = []; getEx = True
for i in trainingset:
    blurr = cv2.GaussianBlur(i,(5,5),0)#getting only green part of the image to compare
    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
    #GREEN PARAMETERS
    lower = (25,40,50)
    upper = (75,255,255)
    mask = cv2.inRange(hsv,lower,upper)
    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
    boolean = mask>0
    new = np.zeros_like(i,np.uint8)# conversion to binary format or adding dimension
    new[boolean] = i[boolean]
    new_train.append(new)
    
#    if getEx:
#        plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL
#        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED
#       plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED
#        plt.subplot(2,3,4);plt.imshow(mask) # MASKED
#        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED
#        plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE
#        plt.show()
#        getEx = False
new_train = np.asarray(new_train)

# CLEANED IMAGES
#for i in range(12):
#    plt.subplot(3,4,i+1)
#    plt.imshow(new_train[i])
    
    
labels = preprocessing.LabelEncoder()
labels.fit(traininglabels[0])
print('Classes'+str(labels.classes_))
encodedlabels = labels.transform(traininglabels[0])
clearalllabels = np_utils.to_categorical(encodedlabels)
classes = clearalllabels.shape[1]
print(str(classes))
#traininglabels[0].value_counts().plot(kind='pie')


new_train = new_train/255 #(0-255) ranges are convert to (0.0-1.0) ranges
x_train,x_test,y_train,y_test = train_test_split(new_train,clearalllabels,test_size=0.1,random_state=seed,stratify=clearalllabels)


np.random.seed(seed)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))
#the number of output filter is 64 for this layer
#kernel_size is the size of the CONVOLUTION WINDOW 5 by 5 here
#we can also give strides as a parameter for jumps
#input_shape is a 4D tensor with shape as (batch, row, col, channels)
#kernal and bias initializers
model.add(BatchNormalization(axis=3)) #only along the features that is why axis 3
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

generator = ImageDataGenerator(rotation_range = 180,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,vertical_flip = True)
generator.fit(x_train)

train_generator = generator.flow(x_train, y_train, batch_size=75)

#model = ...  # Get model (Sequential, Functional Model, or Model subclass)

# SETTING UP CHECKPOINTS, CALLBACKS AND REDUCING LEARNING RATE
#lrr = ReduceLROnPlateau(#reduces learning rate...
#                        monitor='val_acc', #what is needed to be monitor acc in this case as we want to maximize our accuracy
#                        patience=3, #patience(wait for 3 epochs) if no progress then do
#                        verbose=1, 
#                        factor=0.4, #factor by which LR will be reduces new_lr = lr*factor 
#                        min_lr=0.00001)#stops if LR reaches this point
filepath="E:/DELHI(sonipat)(TRAINING)/CLASSIFICATION OF PLANT SEEDLING/plant-seedlings/weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoints = ModelCheckpoint(filepath, monitor='val_acc', 
                              verbose=1, save_best_only=True, mode='max')
#filepath="E:/DELHI(sonipat)(TRAINING)/CLASSIFICATION OF PLANT SEEDLING/plant-seedlings/weights.last_auto4.hdf5"
#checkpoints_full = ModelCheckpoint(filepath, monitor='val_acc', 
#                                 verbose=1, save_best_only=False, mode='max')
#
callbacks_list = [checkpoints, lrr, checkpoints_full]

#MODEL
model.fit_generator(train_generator, epochs=35, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0], callbacks=callbacks_list)

model.save("E:/DELHI(sonipat)(TRAINING)/CLASSIFICATION OF PLANT SEEDLING/plant-seedlings/my_model.h5")

# LOADING best weight to MODEL
model.load_weights("E:/DELHI(sonipat)(TRAINING)/CLASSIFICATION OF PLANT SEEDLING/weights.best_17-0.96.hdf5")#custom weights to give changing weights

#dataset = np.load("C:/Users/admin/Desktop/Data.npz") #custom data
#data = dict(zip(("x_train","x_test","y_train", "y_test"), (dataset[k] for k in dataset)))#changing to zip file and then mapping with dic
#x_train = data['x_train']
#x_test = data['x_test']
#y_train = data['y_train']
#y_test = data['y_test']
    
print(model.evaluate(x_train, y_train))  # Evaluate on train set
print(model.evaluate(x_test, y_test))  # Evaluate on test set

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
cmatrix =confusion_matrix(y_test,y_pred)
print(cmatrix)




#..............................FINAL TESTING DEPARTMENT...............................................


path_to_test = 'E:/DELHI(sonipat)(TRAINING)/CLASSIFICATION OF PLANT SEEDLING/plant-seedlings/TEST IMAGES/*.png'
pics = glob(path_to_test)

testimages = []
tests = []
count=1
num = len(pics)

for i in pics:
    print(str(count)+'/'+str(num),end='\r')
    tests.append(i.split('\\')[1])
    testimages.append(cv2.resize(cv2.imread(i),(scale,scale)))
    count = count + 1

testimages = np.asarray(testimages)

newtestimages = []
sets = []
getEx = True
for i in testimages:
    blurr = cv2.GaussianBlur(i,(5,5),0)
    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
    
    lower = (25,40,50)
    upper = (75,255,255)
    mask = cv2.inRange(hsv,lower,upper)
    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
    boolean = mask>0
    masking = np.zeros_like(i,np.uint8)
    masking[boolean] = i[boolean]
    newtestimages.append(masking)
    
    if getEx:
        plt.subplot(2,3,1);plt.imshow(i)
        plt.subplot(2,3,2);plt.imshow(blurr)
        plt.subplot(2,3,3);plt.imshow(hsv)
        plt.subplot(2,3,4);plt.imshow(mask)
        plt.subplot(2,3,5);plt.imshow(boolean)
        plt.subplot(2,3,6);plt.imshow(masking)
        plt.show()
        getEx=False

newtestimages = np.asarray(newtestimages)
# OTHER MASKED IMAGES
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(newtestimages[i])
    
newtestimages=newtestimages/255
prediction = model.predict(newtestimages)

# PREDICTION TO A CSV FILE
pred = np.argmax(prediction,axis=1)#CONVERT TO ARRAY
predStr = labels.classes_[pred]#BINARY CLASS TO STRING CLASS CONVERSION
result = {'file':tests,'species':predStr}#ORIGINAL ID LINKED TO PREDICTED STRING
result = pd.DataFrame(result)
result.to_csv(r"E:\DELHI(sonipat)(TRAINING)\CLASSIFICATION OF PLANT SEEDLING\plant-seedlings\test_Prediction.csv",index=False)





