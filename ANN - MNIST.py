# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from sklearn import metrics

#Load the MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images shape: ',train_images.shape)
print('test_images shape: ',test_images.shape)
print('train_labels shape: ',train_labels.shape)

print(train_images)

#Process the data for the usage of ANN
train_images=train_images.reshape((60000, 28*28)) #Flatten the image
print(train_images.shape)
train_images=train_images.astype('float32')/255
print(train_images)
test_images=test_images.reshape((10000, 28*28))
print(test_images.shape)
test_images=test_images.astype('float32')/255
print(test_images)

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

print(train_labels)
print(test_labels)

#Define the network
network=models.Sequential()
network.add(layers.Dense(10,activation='softmax',input_shape=(28*28,)))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


print(network.summary())

#Train the network
network.fit(train_images,train_labels,epochs=5,batch_size=128,verbose=2)

#Evaluate the network
test_loss, test_acc = network.evaluate(test_images,test_labels)
print('test acc:', test_acc)

#Make prediction
print(test_images.shape)
x=test_images[1]
print(x.shape)
y=test_labels[1]

prediction=network.predict(x.reshape(1,28*28))
print("%s predicted: %s" %(y,prediction))
print("%s predicted: %s" %(np.argmax(y), prediction.argmax()))


predictions=network.predict(test_images)

test_labels=np.argmax(test_labels,axis=1)
pred_labels=np.argmax(predictions,axis=1)

print("truth: %s\npredicted: %s" %(test_labels,pred_labels))

print("Classification report for classifier %s: \n%s\n" %(network, metrics.classification_report(test_labels,pred_labels)))

print('Confusion Matrix')
print(metrics.confusion_matrix(test_labels,pred_labels))












