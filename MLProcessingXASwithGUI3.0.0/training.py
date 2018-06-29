#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     Training
#==============================================================================
import mlAPI as api
nn=api.NeuralNetwork()
svm=api.SVM()
knn=api.KNeighbors()

#default=DNN
com_4_model_state="DNN"
evaluation_cv_fold=0
#evaluation_result="null"
global model

def train():
    if  com_4_model_state == "DNN":
        #DNN
        global output_accuracy
        output_accuracy=nn.training()
        #保存模型，便于预测时使用
        global model
        model=nn.classifiers()
#        print(output_accuracy)
#        nn.evaluation(K_Fold=5)
    elif com_4_model_state == "SVM":
        #SVM
        output_accuracy=svm.training()
        svm.evaluation(K_Fold=5)
        
    elif com_4_model_state =="KNN":
        #KNN
        output_accuracy=knn.train_accuracy()
        knn.evaluation(K_Fold=5)
    return 0

def evaluation():
    if  com_4_model_state == "DNN":
        #DNN
        evaluation_result=nn.evaluation(K_Fold=evaluation_cv_fold)
        
    elif com_4_model_state == "SVM":
        #SVM
        svm.evaluation(K_Fold=evaluation_cv_fold)
        
    elif com_4_model_state =="KNN":
        #KNN
        knn.evaluation(K_Fold=evaluation_cv_fold)
    return evaluation_result
    

if __name__=='__main__':
    train()
