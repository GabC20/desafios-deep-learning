# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:27:11 2019

@author: Gabriel Carvalho
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


# Definindo diretórios que contém as imagens e as classes para base de dados de testes

TEST_DATADIR = "D:\Coisas\Desafio CyberLabs\Desafio Visao Computacional\Test_Data"
CATEGORIES = ["AZUL", "GOL"]
IMG_SIZE = 64


# Processamento das imagens das classes

testing_data = []

def process_testing_data():
    for category in CATEGORIES:
        path = os.path.join(TEST_DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
            except Exception as e:
                pass
            
process_testing_data()


### Não é necessário colocar as imagens em ordem aleatória já que apenas precisamos classificar essas imagens 


# Criando vetores para armazenar para armazenar as imagens vetorizadas e suas classificações (como AZUL  ou GOL)

X_testing = []
y_testing = []


for features, label in testing_data:
    X_testing.append(features)
    y_testing.append(label)
    
    
# Transformando os vetores X_train e y_train em vetores numpy
    
X_testing = np.array(X_testing).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_testing = np.array(y_testing)


# Salvando os vetores de dados

pickle_out = open("X_testing.pickle", "wb")
pickle.dump(X_testing, pickle_out)
pickle_out.close()

pickle_out = open("y_testing.pickle", "wb")
pickle.dump(y_testing, pickle_out)
pickle_out.close()




















