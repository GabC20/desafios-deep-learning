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


# Testando imagens do conjunto de teste

for category in CATEGORIES:
    path = os.path.join(TEST_DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break






















