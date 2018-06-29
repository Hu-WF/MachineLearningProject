#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#------------------------------------------------------------------------------
# Author:      HuWeiFeng
# Created:     2018-05-02
# Finished:    2018-05-06
# E-mail:      674649741@qq.com
# Purpose:     绘图。共8张图片。
#------------------------------------------------------------------------------
from MLFunctions import Drawing
from MLDataProcessing import global_class_num

def main():
    dr=Drawing()
    #绘制背景图和样本图，共1+4张
    dr.draw_back_and_samples()
    #绘制PCA降维后的二维、三维、散点图，共3张
    dr.draw_pca_2D(global_class_num)
    dr.draw_pca_3D(global_class_num)
    dr.draw_pca_scatter()
    return 0

if __name__=="__main__":
    main()