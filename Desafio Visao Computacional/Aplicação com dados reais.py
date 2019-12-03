# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:16:14 2019

@author: Gabriel Carvalho
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import os


# Definindo o diretório que contém as imagens a serem testadas e o tamanho da imagem redimensionada

real_data = []
REAL_DATADIR = "D:\Coisas\Desafio CyberLabs\Desafio Visao Computacional\Real_Data"
IMG_SIZE = 64


# Definindo a função que redimensiona a imagem e a salva como vetor

def process_testing_data():
    path = REAL_DATADIR
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        real_data.append([new_array])
            
process_testing_data()


# Colocando as informações de valores de pixel no vetor X_real e normalizando

X_real = []
for features in real_data:
    X_real.append(features)
    
X_real = np.array(X_real).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X_real = X_real/255.


# carregando o modelo

model = tf.keras.models.load_model("Alt Conv Net Desafio CyberLabs")


# Carregando a predição do modelo para esse conjutno de imagens

model_out = model_out = model.predict([X_real])


# Exibindo as imagens e suas predições

path = "D:\Coisas\Desafio CyberLabs\Desafio Visao Computacional\Real_Data"

i = 0

for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        if(model_out[i][0] < 0.5):
            print("                    Azul")
        else:
            print("                    Gol")
        i = i + 1




