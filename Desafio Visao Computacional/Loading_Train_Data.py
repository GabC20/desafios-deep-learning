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

X_train = []
y_train = []

for features, label in training_data:
    X_train.append(features)
    y_train.append(label)
 
    
# Transformando os vetores X_train e y_train em vetores numpy 
    
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array(y_train)









