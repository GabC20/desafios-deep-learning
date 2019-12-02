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
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([new_array, class_num])
            
process_testing_data()




















