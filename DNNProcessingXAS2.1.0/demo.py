#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-19
# Version:     2.0.0
# E-mail:      674649741@qq.com
# Purpose:     分析数据时可以临时画出指定csv的指定列数据
#==============================================================================
from mlAPI import Drawing
dr=Drawing()

#draw_specific_col:{输入文件；保存图片名称；绘制哪几列(数量不限)}
dr.draw_specific_col("file_temp\\sample_3_debacked.csv","test",1,2,3,4,5)