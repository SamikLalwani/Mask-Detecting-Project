
"""
AI with python:
Mask detecting project

"""


import os
get_ipython().system('pip3 install opencv-python')
import cv2

Features=[]
Target=[]
for i in range(43):
    ImageNames=os.listdir("-----" +"/"+ str(i)) # Replace ----- with file path
    for name in ImageNames:
        ImageAsArray = cv2.imread("-----" +"/"+ str(i) +"/"+ name)
        Features.append(ImageAsArray)
        Target.append(i)
    print("Inside folder:",i)


import numpy as np
Features=np.array(Features)
Target=np.array(Target)
Features.shape
Target.shape


from sklearn.model_selection import train_test_split
train_features,test_features,train_target,test_target=train_test_split(Features,Target,test_size=0.2)

train_features.shape
train_target.shape
test_features.shape


def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img/255
    return img


train_features=list(map(preprocessing,train_features))
test_features=list(map(preprocessing,test_features))
train_features=np.array(train_features)
test_features=np.array(test_features)
train_features.shape
test_features.shape


train_features=train_features.reshape(27839, 32, 32,1)
test_features=test_features.reshape(6960, 32, 32,1)


from keras.preprocessing.image import ImageDataGenerator
dataGen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(train_features)
batches=dataGen.flow(train_features,train_target,batch_size=20)
len(batches)


batches
X_batch,y_batch=next(batches)
X_batch.shape
y_batch.shape


import matplotlib.pyplot as plt
for i in range(15):
    plt.subplot(5,4,i+1)
    plt.imshow(X_batch[i].reshape(32,32))
plt.show


train_target.shape
from keras.utils.np_utils import to_categorical
train_target=to_categorical(train_target,43)
test_target=to_categorical(test_target,43)


import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout


model=Sequential()
model.add(Conv2D(600,(2,2),input_shape=(32,32,1),activation="relu"))
model.add(Conv2D(600,(2,2),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(300,(2,2),activation="relu"))
model.add(Conv2D(300,(2,2),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(300,activation="relu"))
model.add(Dense(43,activation="softmax"))


from keras.optimizers import Adam
model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.fit_generator(dataGen.flow(train_features,train_target,batch_size=20),epochs=20)


import cv2
import numpy as np


cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,200)


while True:
    success,imgOriginal=cap.read()
    imp=np.asarray(imgOriginal)
    img=cv2.resize(img,(32,32))
    img=preprocessing(img)
    img=img.reshape(1,32,32,1)
    cv2.putText(imgOriginal,"Class: ",(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(imgOriginal,"Probability: ",(20,75),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    predictions=model.predict(img)
    classIndex=model.predict_classes(img)
    probabilityValue=np.amax(predictions)
    
    
    if probabilityValue > 0.75:
        cv2.putText(imgOriginal,getClassName(classIndex),(120,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(imgOriginal,str(probabilityValue)+str(%),(120,75),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("Result",imgOriginal)




