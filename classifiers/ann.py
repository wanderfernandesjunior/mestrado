# Rede Neural (encapsulado dentro da classe base do scikit-learn) conforme dica do Prof. Boltd

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

class RedeNeural(BaseEstimator, ClassifierMixin):
    """ Classe customizada com base no sklearn """
    def __init__(self):
        pass
  
    def fit(self, X_train, y_train):
        """ Estrutura da rede neural adaptada de Kowsari (2019) """
        nFeatures = X_train.shape[1]
        nClasses = np.unique(y_train).shape[0]
        print(f'nFeatures: {nFeatures} | nClasses: {nClasses}')
        nNos = 256
        nCamadas = 4
        dropout=0.25

        self.model = tf.keras.models.Sequential()    

        self.model.add(tf.keras.layers.Dense(nNos,input_dim=nFeatures,activation='relu'))
        self.model.add(tf.keras.layers.Dropout(dropout))

        for _ in range(0, nCamadas):
            self.model.add(tf.keras.layers.Dense(nNos,input_dim=nNos,activation='relu'))
            self.model.add(tf.keras.layers.Dropout(dropout))

        #self.model.add(tf.keras.layers.Dense(nClasses, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(1, activation='tanh'))
        
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])       

        self.model.fit(X_train, y_train, epochs=5, verbose=2)

    def predict(self, X_test):
        
        predictions = self.model.predict(X_test)
        #print(f'\nann prediction argmax {np.argmax(predictions,axis=1)}\n')
        #return np.argmax(predictions,axis=1)
        return predictions