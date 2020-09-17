# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:22:39 2020

@author: Koray
"""
import pandas as pd 

data=pd.read_csv("CarPrice.csv")

#%% Veri Önişleme

data=data.iloc[:,3:]

#%% Price sütunun alınması

doornumber=data.iloc[:,2]

#%% Bazı sutunların sılınmesı

data=data.drop(['doornumber','enginetype','fuelsystem',
                'enginelocation','cylindernumber'],axis=1)

#%% Kategorik Veri > Numeric Veri

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data.iloc[:,0]=le.fit_transform(data.iloc[:,0])

data.iloc[:,1]=le.fit_transform(data.iloc[:,1])

data.iloc[:,2]=le.fit_transform(data.iloc[:,2])

data.iloc[:,3]=le.fit_transform(data.iloc[:,3])

#%% Dummy Variebles- One Hot Encoder

data=pd.get_dummies(data,columns=['aspiration','fueltype',
                                  'carbody','drivewheel'],
                    prefix=['aspiration','fueltype',
                            'carbody','drivewheel'])

#%% Door Number datasının Numerıc yapılması

doornumber=le.fit_transform(doornumber)


#%% Train ve Test Datalarının Ayrılması

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,doornumber,
                                               test_size=0.33,
                                               random_state=0)

#%% Datamızı Standartlaştırıyoruz

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

x_train=ss.fit_transform(x_train)

x_test=ss.fit_transform(x_test)

#%% Sinir Ağımızı Oluşturuyoruz

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping,ModelCheckpoint

model=Sequential()

model.add(Dense(units=16,activation='relu',input_dim=26))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'])

#Overfitting'i engellemek için EarlyStopping kullandık.
#En iyi modeli kaydetmesi için de ModelCheckpoint kullandık.

callbacks=[EarlyStopping(monitor='val_loss',patience=2,verbose=1,
                         mode='auto'),
           ModelCheckpoint(filepath='model.h5',monitor='val_loss',
                           save_best_only=True)]

history=model.fit(x_train,y_train,callbacks=callbacks,epochs=20,
          verbose=1)


#%% Predict

y_pred=model.predict(x_test)

y_pred=(y_pred > 0.5)

#%% Confusion Matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)














