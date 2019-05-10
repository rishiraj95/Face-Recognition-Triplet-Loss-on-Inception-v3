"""
Train Siamese Neural Networks on feature embeddings
"""

import os
import json
import random
import numpy as np
import h5py

input_size=8000
file_list = os.listdir('jsonResp')
input_vecs=np.zeros((input_size*2,1024))
labels=np.zeros((input_size*2,1))
cntr=0
i=0

#Get the embeddings and labels in a input vector
while i <=input_size:
    
    with open('jsonResp/'+file_list[i]) as f:
        face1_json=json.load(f)
    face1_vec=np.array(face1_json['outputs'][0]['data']['regions'][0]['data']['embeddings'][0]['vector'])
    for j in range(i+1,i+9):
        try:
            with open('jsonResp/'+file_list[j]) as f:
                face2_json=json.load(f)
            face2_vec=np.array(face2_json['outputs'][0]['data']['regions'][0]['data']['embeddings'][0]['vector'])
            input_vecs[cntr,:]=abs(face1_vec-face2_vec)
            if file_list[i][:-10]==file_list[j][:-10]:
                labels[cntr]=1
            
            cntr+=1
        except:
            continue
        
    while file_list[i][:-10]==file_list[j][:-10]:
        j+=1
    for k in random.sample(list(range(i))+list(range(j,input_size+1000)),8):
        try:
            with open('jsonResp/'+file_list[k]) as f:
                face2_json=json.load(f)
            face2_vec=np.array(face2_json['outputs'][0]['data']['regions'][0]['data']['embeddings'][0]['vector'])
            input_vecs[cntr,:]=abs(face1_vec-face2_vec)
            if file_list[i][:-10]==file_list[k][:-10]:
                labels[cntr]=1
            
            cntr+=1
        except:
            continue
        
    i=j

#Shuffle and split data into train and test
input_vecs=input_vecs[:cntr+1]
labels=labels[:cntr+1]
shuff_ind=np.arange(len(labels))
np.random.shuffle(shuff_ind)
labels=labels[shuff_ind]
input_vecs=input_vecs[shuff_ind]

labels_train=labels[:int(0.85*len(labels))]
labels_test=labels[int(0.85*len(labels))+1:]
input_vecs_train=input_vecs[:int(0.85*len(labels))]
input_vecs_test=input_vecs[int(0.85*len(labels))+1:]


#Build model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model=Sequential()
model.add(Dense(800,input_shape=(1024,),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#Train, test and save model
from keras.optimizers import SGD
sgd=SGD(lr=0.03,decay=0.0001,momentum=0.5,nesterov=False)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(input_vecs_train,labels_train,batch_size=1,epochs=15)
model.evaluate(input_vecs_test,labels_test)

model.save('verify_face.h5')




