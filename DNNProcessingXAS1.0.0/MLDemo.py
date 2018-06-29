#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#------------------------------------------------------------------------------
# Author:      HuWeiFeng
# Created:     2018-04-24
# Finished:    2018-05-08
# E-mail:      674649741@qq.com
# Purpose:     用于通过画图做临时数据分析、观察；可画出指定文件的指定列光谱图像。不参与正式分析
#------------------------------------------------------------------------------
#from MLFunctions import CreateFolder
from MLFunctions import Drawing
#from MLFunctions import CSVPreProcessing
#from MLDataProcessing import global_class_num

def main():
#    cf=CreateFolder()
    dr=Drawing()
    #cpp=CSVPreProcessing()
    #建立存储这些临时的分析图片和数据的文件夹，文件夹名保持和draw_specific_col函数里的一样即可
#    cf.create_folder("temp_graph")
#    temp_file_path="temp_file"
#    cf.create_folder(temp_file_path)
    
    #绘制多种CT谱，比较那种重建效果较好
#    dr.draw_specific_col("temp_file\\ART.csv","ART",1,2,3,4,5,6,7,)
#    dr.draw_specific_col("temp_file\\ART_HC.csv","ART_HC",1,2,3,4,5,6,7,)
    
#    dr.draw_specific_col("temp_file\\SART.csv","SART",1,2,3,4,5,6,7,)
#    dr.draw_specific_col("temp_file\\SART_HC.csv","SART_HC",1,2,3,4,5,6,7,)
    
#    dr.draw_specific_col("temp_file\\sl.csv","sl",1,2,3,4,5,6,7,)
#    dr.draw_specific_col("temp_file\\sl_HC.csv","sl_HC",1,2,3,4,5,6,7,)
    

#    dr.draw_specific_col("file_ct\\ct_cut_data.csv","2CT原始图",1,2,3,4,5,6,7,)
#    
##    dr.draw_specific_col("file_ct\\ct_denoise_data.csv","3CT降噪后的图",1,2,3,4,5,6,7,)
#    
#    dr.draw_specific_col("file_ct\\ct_scaled_data.csv","4CT先降噪再标准化后的图",1,2,3,4,5,6,7,)
#    
##    dr.draw_specific_col("temp_file\\afds.csv","原始谱去噪后",1,101,201,301,401,501,601)
#    dr.draw_specific_col("file\\sample_cut_data.csv","5测得原始谱去背景前",1,101,201,301,401,501,601)
#    dr.draw_specific_col("file\\sample_debacked_data.csv","7测得原始谱去背景后",1,101,201,301,401,501,601,701,801,901)
#    dr.draw_specific_col("file\\sample_scaled_data.csv","8测得原始谱归一化后",1,101,201,301,401,501,601,701,801,901) 
#    
##    dr.draw_specific_col("temp_file\\load_back_data.csv","CT背景",2,3,4,5,6,7,)


    #画加噪声后的图
    
#    dr.draw_specific_col("noise\\SG.csv","1SG原始谱",1,2,3) 
#    dr.draw_specific_col("noise\\TAR.csv","1TAR原始谱",1,2,3) 
#    dr.draw_specific_col("noise\\UV.csv","1UV原始谱",1,2,3,4) 
#    
#    dr.draw_specific_col("noise\\SG_debacked.csv","2SG原始谱裁剪去背景后",1,2,3)
#    dr.draw_specific_col("noise\\SG_scaled.csv","B_SG原始谱裁剪去背景归一化后",1,2,3)
#    dr.draw_specific_col("noise\\UV_scaled.csv","B_UV原始谱裁剪去背景归一化后",1,2,3)
#    dr.draw_specific_col("noise\\TAR_scaled.csv","B_TAR原始谱裁剪去背景归一化后",1,2,3)
#    dr.draw_specific_col("file\\sample_scaled_mean_data.csv","E_样本",1,2,3,4,5,6,7,8,9,10)
    
    
    dr.draw_specific_col("file\\sample_scaled_mean_data.csv","E_所有样本",1,2,3,4,5,6,7,8,9,10)
    dr.draw_specific_col("file\\sample_scaled_mean_data.csv","E_样本",2,3,5,9)
    
    
#    dr.draw_specific_col("noise\\ART_HC.csv","C_ART_HC",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("noise\\ART_HC_cut.csv","6ART_HC裁剪后",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_init_data_denoised.csv","D_CT10_初始降噪",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_scaled_data.csv","D_CT10降噪归一化后",1,2,3,4,5,6,7,8,9,10) 
    
    
    dr.draw_specific_col("file_ct\\ct_scaled_data.csv","D_CT10降噪归一化后，识别不出的四个",2,3,5,9) 
    dr.draw_specific_col("file_ct\\ct_scaled_data.csv","D_CT10降噪归一化后",1,2,3,4,5,6,7,8,9,10) 
    
    
    dr.draw_specific_col("testing\\B.csv","F_再重建",1,) 
    dr.draw_specific_col("testing\\ART_HC.csv","F_ART_HC",1,) 
    dr.draw_specific_col("testing\\SART_HC.csv","F_SART_HC",1,) 
    dr.draw_specific_col("testing\\before_scale_unknown_algorithm.csv","F归一化前，未知重建算法",1,2,3,4,5,6,7,8,9,10) 
    
    
#画不同阶数的小波降噪  
#    dr.draw_specific_col("file_ct\\ct_cut_data_denoisedN1.csv","test1",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_cut_data_denoisedN2.csv","test2",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_cut_data_denoisedN3.csv","test3",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_cut_data_denoisedN5.csv","test5",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_cut_data_denoisedN10.csv","test10",1,2,3,4,5,6,7,8,9,10) 
    
#    dr.draw_specific_col("file_ct\\ct_scaled_data_denoised.csv","D_CT10归一化后再降噪",1,2,3,4,5,6,7,8,9,10) 
#    dr.draw_specific_col("file_ct\\ct_scaled_data.csv","D_CT10归一化后",10,) 
#    dr.draw_specific_col("noise\\ART_HC_cut.csv","6ART_HC裁剪后",10) 
#
#    dr.draw_specific_col("noise\\after_noised_SG.csv","4SG加噪声后",1,10,20,30,40,50,60,70) 
#    dr.draw_specific_col("noise\\after_noised_TAR.csv","4TAR加噪声后",1,10,20,30,40,50,60,70) 
#    dr.draw_specific_col("noise\\after_noised_UV.csv","4UV加噪声后",1,10,20,30,40,50,60,70) 
    
    return 0

if __name__=="__main__":
    main()