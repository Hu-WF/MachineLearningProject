#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     分析数据时可以临时画出指定csv的指定列数据
#==============================================================================
from mlAPI import Drawing
dr=Drawing()

from mlAPI import CSVProcessing
cp=CSVProcessing()

from mlFunctions import MLCsvProcessing
mcp=MLCsvProcessing()

#draw_specific_col:{输入文件；保存图片名称；绘制哪几列(数量不限)}
#dr.draw_specific_col("file_temp\\sample_3_debacked.csv","test",1,2,3,4,5)

#mcp.get_back_smart([1,10],[11,20])

#import pandas as pd
#import os
#
#a={}
#a=pd.DataFrame(a)
#a.to_csv("file_temp\\testing.csv",header=None,index=None)
#
#if os.path.exists("file_temp\\testing.csv"):
#    print("YES")
#    os.remove("file_temp\\testing.csv")

cp.split_combine_data("file_output\\load_data.csv","file_temp\\temp.csv",[1,10],[11,20])