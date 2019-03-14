#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     Training
#==============================================================================
"""
①.对于端到端的模型（如没有经过PCA的DNN模型，CNN模型等），始终需要开启transpose_data=True来手动转置data；
②.对于以经过PCA、SAE降维的模型，则不必开启；
    由于①情况较为常见，因此默认参数均按此设置；对于②需更改默认参数

"""
import mlAPI as api
import dlAPI

#=============================================================================
#DNN,默认包含PCA降维操作，data已转置好，无需再转置
"""PCA-DNN,SAE-DNN"""
#nn=api.NeuralNetwork()
#nn.training()#transpose_data默认True
#nn.evaluation(K_Fold=10,)#transpose_data默认False

#SVM
#svm=api.SVM()
#svm.training()
#svm.evaluation(K_Fold=10)

#KNN
#kNN=api.KNeighbors()
#kNN.training()
#kNN.evaluation(K_Fold=10)


#==============================================================================
#CNN
"""1D-CNN"""
#暂未添加cross_validation功能，因此cv=False
cnn=dlAPI.CNNModel()
cnn.training(transpose_data=True)
cnn.evaluation(K_Fold=3,cv=False)#transpose_data是设置cross_validation的参数，因此这里不必设置


#==============================================================================
#DNN，若原始data没有经过PCA降维，而直接作为输入，则需要开启转置
"""端到端-DNN(原始光谱作为输入，不做任何特征提取处理)"""
#nn=api.NeuralNetwork()
#nn.training(transpose_data=True)#默认True
#nn.evaluation(K_Fold=10,transpose_data=True)#默认False

