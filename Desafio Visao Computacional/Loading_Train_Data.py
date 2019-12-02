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


# Teste para verificar se foi possível achar o caminho especificado

for category in CATEGORIES:
    # caminho para os diretórios com as imagens da AZUL e da GOL
    path = os.path.join(TRAIN_DATADIR, category)  
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break


