#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#------------------------------------------------------------------------------
# Author:      HuWeiFeng
# Created:     2018-04-24
# Finished:    2018-05-11
# E-mail:      674649741@qq.com
# Purpose:     用于通过去背景、标准化、降维等一系列处理，生成CT_x_data.csv用于预测。
#------------------------------------------------------------------------------
import MLFunctions as f
import pandas as pd

#输入值（默认为已经去背景后的值）：
#目前用的ct_init_data.csv是归一化前未知重建算法.csv
ct_initial_data="\\ct_init_data.csv"
#ct_initial_data="\\ct_init_data_denoised.csv"

#剪切后：
ct_cut_data="\\ct_cut_data.csv"

##去背景后：
#ct_deback_data="\\ct_deback_data.csv"

#降噪后：(可不降噪)
ct_denoised_data="\\ct_denoise_data.csv"

#归一化或标准化后：
ct_scaled_data="\\ct_scaled_data.csv"

#降维后
ct_dre_data="\\ct_dre_data.csv"


#最终的用于预测的文件
ct_x_data="\\ct_x_data.csv"


#所有生成CT_x_data.csv的操作都在一个单独的文件夹内完成
folder_name="file_ct"
cf=f.CreateFolder()
cf.create_folder(folder_name)

ttc=f.TxtToCSV()
#ttc.txt_to_csv("\\重建光谱back")

#==============================================================================
cpp=f.CSVPreProcessing()

#剪切
#cpp.cut_line(50,250,folder_name+ct_initial_data,folder_name+ct_cut_data)
#降噪：这部分在MATLAB中完成，也可不做：若不做：
#ct_denoised_data=ct_cut_data#执行这句话意味着没有做降噪处理
#ct_denoised_data="\\ct_cut_data_denoisedN3.csv"
ct_denoised_data="\\ct_cut_data_denoisedjia.csv"

#归一化：
cpp.data_normalization(folder_name+ct_denoised_data,folder_name+ct_scaled_data,scale_to_01=True)
#cpp.data_normalization(folder_name+ct_denoised_data,folder_name+ct_scaled_data,scale_to_01=False)

#归一化：（和原始数据同步归一化）
#ttc.combine_csv("file\\sample_debacked_data.csv",folder_name+ct_denoised_data,"file_ct\\combine_for_scale.csv")
#cpp.data_normalization("file_ct\\combine_for_scale.csv","file_ct\\combine_and_scaled.csv",scale_to_01=True)
#cpp.split_data("file_ct\\combine_and_scaled.csv",1001,1010,folder_name+ct_scaled_data)

#----------------------------------------------------
#跳过归一化步骤，对应于已经归一化后的几种重建算法
#ct_scaled_data=ct_denoised_data
#----------------------------------------------------

#降维：（和原始数据同步降维）
##先合并两个待降维文件：
ttc.combine_csv("file\\sample_scaled_data.csv",folder_name+ct_scaled_data,"file_ct\\combine_for_dre.csv")
##再降维
dr=f.DReduction(5)
dr.universal_pca("file_ct\\combine_for_dre.csv","file_ct\\combine_dre_data.csv")

#从ct_dre_data.csv中分离出用于预测的那部分：
#自动计算需要从哪里分离到哪里：
combine_data=pd.read_csv("file_ct\\combine_for_dre.csv",header=None,index_col=None)
combine_num=combine_data.shape[1]
ct_data=pd.read_csv(folder_name+ct_scaled_data,header=None,index_col=None)
ct_num=ct_data.shape[1]
#自动按行分离：
cpp.split_data_by_row("file_ct\\combine_dre_data.csv",combine_num-ct_num+1,combine_num,folder_name+ct_x_data)

