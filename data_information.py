#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     数据集前置信息，包括样本类数、类名、输出文件和图片名称。
#==============================================================================
#GUI参数绑定，传给mlAPI的部分参数===============================================
#样本类数和名称
global_class_num=0
global_class_names='null'
#设定测试集比例，由GUI传入值大小，传出给mlAPI中的ROC，PR，training.
test_set_ratio=0
#传入系统评估时的相关参数
evaluation_botton_state=[1,1,1,1]
#==============================================================================

class SampleInformation():
    """定义数据集前置属性，统一在此处修改"""
    def __init__(self):
#        self.class_num=6
#        self.class_names=['FeiRou','Gan','Shen','Shourou','Wei','Xin']
        self.class_num=global_class_num
        self.class_names=global_class_names
#        self.class_names=['Normal','CancerA','CancerB']
#        self.class_names=['Normal','Sick']
    #自动根据分类数生成样本名称,串成一个字符串
    def auto_create_name(self,num):
        num=int(num)
        i=0
        str_name=''
        while(i<num):
            i+=1
            str_name+="S"+str(i)+","
        return str_name


class FileNameSetting():
    """1.包含数据处理过程中所有需存储的.csv文件名，分并行和串行两层级命名"""
    def __init__(self):
#=============================1.文件夹名称======================================
        self.folder_1="file_output\\"#输出数据文件夹，串行
        self.folder_2="file_temp\\"#每一步运行时的临时存储文件夹，并行
        self.folder_3="graph_output\\"#输出图像文件夹
        self.folder_4="graph_demo\\"#临时绘图观察文件夹
        self.folder_5="file_predict\\"#用于预测的相关文件夹
#==========================2.输出串行文件名(必要)===============================
        #放于folder_1:file_output
        self.load_data_output=self.folder_1+"load_data.csv"#输出原始数据
        self.load_predict_data_output=self.folder_5+"load_predict_data.csv"#输出原始待预测
        self.sample_data_output=self.folder_1+"sample.csv"#输出最终样本数据
        self.back_data_output=self.folder_1+"back.csv"#输出最终背景数据
        self.label_data_output=self.folder_1+"label.csv"#输出标签
        self.predict_sample_data_output=self.folder_5+"predict_sample.csv"
        #保存一份降维前的sampledata，为了同predict data协同进行降维，放在文件夹1是因为此时还未生成folder_5
        self.sample_data_before_dre=self.folder_1+"sample_data_before_dre.csv"
#=========================3.输出并行文件名(可选操作)=============================
        #3.1txt转csv
        #3.2分割csv       
        self.back_data=self.folder_2+"back_1.csv"
        self.sample_data=self.folder_2+"sample_1.csv"
        self.predict_sample_data=self.folder_5+"predict_sample_1.csv"
        #3.3裁剪csv
        self.back_cut_data=self.folder_2+"back_2_cut.csv"
        self.sample_cut_data=self.folder_2+"sample_2_cut.csv"
        self.predict_sample_cut_data=self.folder_5+"predict_sample_2_cut.csv"
        #3.4均值
        self.back_mean_data=self.folder_2+"back_3_mean.csv"
        #3.5去背景
        self.sample_debacked_data=self.folder_2+"sample_3_debacked.csv" 
        self.predict_sample_debacked_data=self.folder_5+"predict_sample_3_debacked.csv"
        #3.6归一化
        self.sample_scaled_data=self.folder_2+"sample_4_scaled.csv"
        self.predict_sample_scaled_data=self.folder_5+"predict_sample_4_scaled.csv"
        #3.7样本求三种均值；仅供画图使用，不参与余下计算过程
        self.sample_cut_mean_data=self.folder_2+"sample_5_cut_mean.csv"
        self.sample_debacked_mean_data=self.folder_2+"sample_6_debacked_mean.csv"
        self.sample_scaled_mean_data=self.folder_2+"sample_7_scaled_mean.csv"    
        #3.8降维
        self.dred_data=self.folder_2+"sample_8_dre.csv"
        self.predict_sample_dred_data=self.folder_5+"predict_sample_8_dre.csv"
        #3.9输出标签
        self.label_data=self.folder_2+"label_1.csv" 
        #3.10采用插值背景去背景时的文件名
        #self.start_back=self.folder_2+"start_back.csv"
        #self.end_back=self.folder_2+"end_back.csv"
        #self.mean_start_back=self.folder_2+"mean_start_back.csv"
        #self.mean_end_back=self.folder_2+"mean_end_back.csv"
        #self.interpolation_back=self.folder_2+"interpolation_back.csv"
        #3.11用于对CT重建结果进行预测
        #self.CT_x_data=self.folder_2+"ct_x_data.csv" 
        #3.12临时存储文件名
        self.temp_data=self.folder_2+"temp_1.csv"


        
        
        
        
        
        
        
        
        
        
        
        
        