#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-19
# Version:     2.0.0
# E-mail:      674649741@qq.com
# Purpose:     Training
#==============================================================================
import mlAPI as api

#DNN
nn=api.NeuralNetwork()
nn.training()
nn.evaluation(K_Fold=5)

#SVM
#svm=api.SVM()
#svm.training()
#svm.evaluation(K_Fold=5)

#KNN
#kNN=api.KNeighbors()
#kNN.train_accuracy()
#kNN.evaluation(K_Fold=5)
