# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:27:25 2019

@author: Gabriel Carvalho
"""

import tensorflow as tf
import pickle


model = tf.keras.models.load_model("Conv Net Desafio CyberLabs")

# Carregando os dados dos dados de teste

pickle_in = open("X_testing.pickle", "rb")
X_testing = pickle.load(pickle_in)
X_testing = tf.cast(X_testing, tf.float32)

pickle_in = open("y_testing.pickle", "rb")
y_testing = pickle.load(pickle_in)


X_testing = X_testing/255.
y_testing = y_testing


# Avaliando o modelo [Loss, Accuracy]



model.evaluate(x=X_testing, y=y_testing, batch_size=32, verbose=0, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

print(model.evaluate(x=X_testing, y=y_testing, batch_size=32, verbose=0, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False))



