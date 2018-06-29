#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#------------------------------------------------------------------------------
# Author:      HuWeiFeng
# Created:     2018-04-24
# Finished:    2018-05-06
# E-mail:      674649741@qq.com
# Purpose:     调用机器学习算法进行Training。
#------------------------------------------------------------------------------
import MLFunctions as func

def main(): 
    #采用MLP
    nn=func.NeuralNetwork()
    nn.training()
    
    #通过K_Fold决定交叉验证折数，default=5，一般5折较常用
    nn.evaluation(K_Fold=5)
    #采用SVM
#    s=func.SVM()
#    s.training()
#    s.evaluation() 
    #采用K近邻
#    k=func.KNeighbors()
#    k.training()
#    k.evaluation()
    return 0

#主程序入口
if __name__=='__main__':
    main()