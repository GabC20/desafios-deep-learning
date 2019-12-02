# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:57:28 2019

@author: Gabriel Carvalho
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


# Definindo diretórios que contém as imagens e as classes

TRAIN_DATADIR = "D:\Coisas\Desafio CyberLabs\Desafio Visao Computacional\Train_Data"
CATEGORIES = ["AZUL", "GOL"]
IMG_SIZE = 64


# Processamento das imagens das classes

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DATADIR, category)  # Caminho para os diretórios com as imagens da AZUL e da GOL
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()


# Testando o tamanho do conjunto de dados

print(len(training_data))


# Colocando todas as imagens do Train Dataset em ordem aleatória

random.shuffle(training_data)


# Criando vetores para armazenar para armazenar as imagens vetorizadas e suas classificações (como AZUL  ou GOL) 

X_training = []
y_training = []

for features, label in training_data:
    X_training.append(features)
    y_training.append(label)
 
    
# Transformando os vetores X_train e y_train em vetores numpy 
    
X_training = np.array(X_training).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_training = np.array(y_training)


# Testando a conversão

print(X_training[0])


# Salvando os vetores de dados 

pickle_out = open("X_training.pickle", "wb")
pickle.dump(X_training, pickle_out)
pickle_out.close()

pickle_out = open("y_training.pickle", "wb")
pickle.dump(y_training, pickle_out)
pickle_out.close()


# Testando se os vetores ficaram salvos com sucesso

pickle_in = open("X_training.pickle", "rb")
X_training = pickle.load(pickle_in)

print(X_training[1])


# testando se X_training e y_training tem o mesmo tamanho

print(len(X_training))
print(len(y_training))











