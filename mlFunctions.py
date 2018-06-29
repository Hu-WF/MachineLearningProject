#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     调用mlAPI类和函数，融合数据data_information进行具体操作。
#==============================================================================
import os
import mlAPI as api
from data_information import FileNameSetting
from data_information import SampleInformation
import data_information as di
import pandas as pd
import re
csvg=api.CSVGenerating()
csvp=api.CSVProcessing()
disi=SampleInformation()
cl=api.CreateLabel()
dr=api.Drawing()


class MLTxt2Csv(FileNameSetting):
    """1.txt转csv操作"""
    def __init__(self,):
        super().__init__()
    #1.1txt to csv    
    def ml_txt_to_csv(self,txt_url):
        csvg.txt_to_csv(txt_url,self.load_data_output)
        return 0
    #1.2导入预测数据，只能单条预测
    def ml_predict_txt_to_csv(self,txt_url):
        csvg.txt_to_csv(txt_url,self.load_predict_data_output)
        return 0
        
    
class MLCsvProcessing(FileNameSetting):
    """2.各子操作并列，即输入为串行文件，输出为串行文件，同时保留备份用于画图"""
    def __init__(self,):
        super().__init__()
#===========================1.split back and sample============================
    #split out back data        
    def get_back_data(self,s_col,e_col):
        csvp.split_csv_by_col(self.load_data_output,s_col,e_col,self.back_data)
        print("Generate "+str(self.back_data)+".")
        #创建外部串行文件
        csvp.create_csv_copy(self.back_data,self.back_data_output)
        return 0
    #split out sample data
    def get_sample_data(self,s_col,e_col):
        csvp.split_csv_by_col(self.load_data_output,s_col,e_col,self.sample_data)
        print("Generate "+str(self.sample_data)+".")
        #创建外部串行文件
        csvp.create_csv_copy(self.sample_data,self.sample_data_output)
        return 0  
    #get predict data       
    def get_predict_sample_data(self,):
        #获取predict data
        csvp.create_csv_copy(self.load_predict_data_output,self.predict_sample_data)
        #创建外部串行文件
        csvp.create_csv_copy(self.predict_sample_data,self.predict_sample_data_output)
        return 0


#=========================2.get back and sample wisely=========================
    #智能化生成back_data：以独立集合形式给定load_data中的任意列，能自动按顺序生成back_data
    #彻底解决实验过程中背景谱顺序散乱问题
    #也可用于提取单个集合中的数据，实现1.split back and sample中的功能。（因此以后可以都用2而不用1）        
    def get_back_data_wisely(self,*columns_sets):
        csvp.split_and_combine_data(self.load_data_output,self.back_data,columns_sets)
        print("Generate "+str(self.back_data)+" from "+str(columns_sets)+" in "+str(self.load_data_output))
        #创建串行文件
        csvp.create_csv_copy(self.back_data,self.back_data_output)
        return 0
    #智能化生成sample_data：以独立集合形式给定load_data中的任意列，能自动按顺序生成sample_data
    #彻底解决实验过程中背景谱顺序散乱问题        
    def get_sample_data_wisely(self,*columns_sets):
        csvp.split_and_combine_data(self.load_data_output,self.sample_data,columns_sets)
        print("Generate "+str(self.sample_data)+" from "+str(columns_sets)+" in "+str(self.load_data_output))
        #创建串行文件
        csvp.create_csv_copy(self.sample_data,self.sample_data_output)
        return 0
    
    #输入类似于"[1,2],[3,4],[5,6]"，将其转换成多个数字列表：[1,2],[3,4],[5,6]
    #用于GUI中的转换
    def convert_str_to_list(self,in_put):
        #提取数字成分
        input_nums=re.findall(r"\d+",in_put)
        #将字符串转成数字
        input_nums=[int(input_num) for input_num in input_nums]
        #提取奇数部分
        odd_nums=input_nums[::2]
        even_nums=input_nums[1::2]
        output_nums=[]
        i=0
        #重新组合奇数和偶数
        for odd_num in odd_nums:
            output_nums.append([odd_nums[i],even_nums[i]])
            i+=1
        return output_nums
        
        
        
     
#=========================2.cut back and sample================================        
    def cut_back_and_sample_line(self,s_row,e_row):
        #裁剪背景谱
        csvp.cut_line(s_row,e_row,self.back_data_output,self.back_cut_data)
        print("Generate "+str(self.back_cut_data)+".")
        #创建裁剪谱串行文件
        csvp.create_csv_copy(self.back_cut_data,self.back_data_output)
        #裁剪样本谱
        csvp.cut_line(s_row,e_row,self.sample_data_output,self.sample_cut_data)
        print("Generate "+str(self.sample_cut_data)+".")
        #创建裁剪谱串行文件
        csvp.create_csv_copy(self.sample_cut_data,self.sample_data_output)
        return 0
    #裁剪预测谱线
    def cut_predict_sample_line(self,s_row,e_row):
        #裁剪背景谱
        csvp.cut_line(s_row,e_row,self.predict_sample_data,self.predict_sample_cut_data)
        print("Generate "+str(self.predict_sample_cut_data)+".")
        #创建裁剪谱串行文件
        csvp.create_csv_copy(self.predict_sample_cut_data,self.predict_sample_data_output)
        return 0
#==============================3.mean back=====================================   
    def mean_one_back(self,):
        csvp.mean_by_line(self.back_data_output,self.back_mean_data)
        print("Generate "+str(self.back_mean_data)+".")
        #创建背景谱均值后串行文件
        csvp.create_csv_copy(self.back_mean_data,self.back_data_output)
        return 0
#====================4.mean back with category number==========================
    #用于分样本类去各自均值背景时，产生各样本各自均值
    def mean_category_back(self,):
        csvp.mean_by_category(self.back_data_output,di.global_class_num,self.back_mean_data)
        print("Generate "+str(self.back_mean_data)+" with category_num = "+str(di.global_class_num))
        #创建串行文件
        csvp.create_csv_copy(self.back_mean_data,self.back_data_output)
        return 0
#=============================5.mean sample====================================
    #此处的样本均值均为画图所用，不参与后续数据处理线程，因此不必生成串行文件
    def mean_all_kinds_sample(self,category):
        #生成各类样本谱线裁剪后的均值
        if os.path.isfile(self.sample_cut_data)==True:
            csvp.mean_by_category(self.sample_cut_data,category,
                                  self.sample_cut_mean_data) 
            print("Generate "+str(self.sample_cut_mean_data)+".")
        else:
            print("There is no file named "+str(self.sample_cut_data)+".")
            
        #生成各类样本谱线去背景后的均值
        if os.path.isfile(self.sample_debacked_data)==True:
            csvp.mean_by_category(self.sample_debacked_data,category,
                                  self.sample_debacked_mean_data)
            print("Generate "+str(self.sample_debacked_mean_data)+".")
        else:
            print("There is no file named "+str(self.sample_debacked_data)+".")
            
        #生成各类样本谱线归一化后的均值
        if os.path.isfile(self.sample_scaled_data)==True:
            csvp.mean_by_category(self.sample_scaled_data,category,
                                  self.sample_scaled_mean_data)
            print("Generate "+str(self.sample_scaled_mean_data)+".")
        else:
            print("There is no file named "+str(self.sample_scaled_data)+".")
        return 0
#====================================6.deback==================================
    #统一去背景
    def deback_one_back(self,):
        csvp.deback(self.sample_data_output,
                    self.back_data_output,
                    self.sample_debacked_data)
        #生成去背景串行文件
        csvp.create_csv_copy(self.sample_debacked_data,self.sample_data_output)
        print("Generate "+str(self.sample_debacked_data)+" based on one_back.")
        return 0
    #分样本类去背景，当每类样本均置于容器，采集各自容器作为背景时用此方法
    def deback_category_back(self,):
        #后面用到再写
        csvp.deback(self.sample_data_output,
                    self.back_data_output,
                    self.sample_debacked_data,
                    category_back=True)
        #生成串行文件
        csvp.create_csv_copy(self.sample_debacked_data,self.sample_data_output)
        print("Generate "+str(self.sample_debacked_data)+" based on category_back.")
        return 0
    #去预测谱线背景
    def predict_deback_one_back(self,):
        csvp.deback(self.predict_sample_data_output,
                    self.back_data_output,
                    self.predict_sample_debacked_data)
        #生成去背景串行文件
        csvp.create_csv_copy(self.predict_sample_debacked_data,self.predict_sample_data_output)
        print("Generate "+str(self.predict_sample_debacked_data)+" based on one_back.")
#==================================7.scale=====================================
    def scale(self,scaled_to_01=False):
        csvp.data_normalization(self.sample_data_output,
                                self.sample_scaled_data,
                                scale_to_01=scaled_to_01)
        #生成归一化串行文件
        csvp.create_csv_copy(self.sample_scaled_data,self.sample_data_output)
        print("Generate "+str(self.sample_scaled_data)+".")
        return 0
    #预测谱线归一化
    def predict_sample_scale(self,scaled_to_01=False):
        csvp.data_normalization(self.predict_sample_data_output,
                                self.predict_sample_scaled_data,
                                scale_to_01=scaled_to_01)
        #生成归一化串行文件
        csvp.create_csv_copy(self.predict_sample_scaled_data,self.predict_sample_data_output)
        print("Generate "+str(self.predict_sample_scaled_data)+".")
        return 0
    
        
class MLCreateLabel(FileNameSetting):
    """3.输出标签label"""
    def __init__(self,):
        super(). __init__()
    #'LabelEncoder' or 'OneHotEncoder',default='LabelEncoder'    
    def create_label(self,category,encode_mode='LabelEncoder'):            
        cl.encoder(self.sample_data,
                   category_num=category,
                   label_csv_name=self.label_data
                   ,encoder_mode=encode_mode)
        #生成标签串行文件
        csvp.create_csv_copy(self.label_data,self.label_data_output)
        print("Generate "+str(self.label_data)+".")
        return 0
    
        
class MLDRe(FileNameSetting):
    """4.降维操作"""
    def __init__(self,):
        super().__init__()
#==============================1.PCA===========================================        
    def myPCA(self,component_num,):
        #拷贝一份降维前的数据，以便于在需要对数据进行预测时，再次协同进行降维
        csvp.create_csv_copy(self.sample_data_output,self.sample_data_before_dre)
        #开始降维
        dre=api.DimensionReduction(components=component_num,
                                   input_data=self.sample_data_output,
                                   dred_data=self.dred_data)
        print("Generate "+str(self.dred_data)+".")
        ret_value=dre.Pca()
        #生成串行文件
        csvp.create_csv_copy(self.dred_data,self.sample_data_output)
        return ret_value
    
    def myPCA_for_predict(self,component_num,):
        #对预测数据进行降维，使用之前保留的那份
        data_set=pd.read_csv(self.sample_data_before_dre,header=None,index_col=None)
        data_set=data_set.T  
        #传入待降维的预测样本
        predict_data=pd.read_csv(self.predict_sample_data_output,header=None,index_col=None)
        predict_data=predict_data.T
        #mlAPI中已经import进PCA，直接调用即可
        pca=api.PCA(n_components=component_num,)
        #相对原模型进行降维，获取转换矩阵（降维模型）
        pca.fit_transform(data_set)
        #将降维模型应用于预测数据
        predict_data=pca.transform(predict_data)
        #输出保存
        predict_data=pd.DataFrame(predict_data)
        predict_data.to_csv(self.predict_sample_dred_data,header=False,index=False)
        #产生串行文件
        csvp.create_csv_copy(self.predict_sample_dred_data,self.predict_sample_data_output)
        return 0

#==============================2.IPCA==========================================        
    def myIPCA(self,component_num):
        dre=api.DimensionReduction(components=component_num,
                                   input_data=self.sample_data_output,
                                   dred_data=self.dred_data)
        print("Generate "+str(self.dred_data)+".")
        dre.IPca()
        #生成串行文件
        csvp.create_csv_copy(self.dred_data,self.sample_data_output)
        return 0
#==============================3.NMF===========================================        
    def myNMF(self,component_num):
        dre=api.DimensionReduction(components=component_num,
                                   input_data=self.sample_data_output,
                                   dred_data=self.dred_data)
        print("Generate "+str(self.dred_data)+".")
        dre.Nmf()
        #生成串行文件
        csvp.create_csv_copy(self.dred_data,self.sample_data_output)
        return 0
#==============================4.TSNE==========================================        
    def myTSNE(self,component_num):
        dre=api.DimensionReduction(components=component_num,
                                   input_data=self.sample_data_output,
                                   dred_data=self.dred_data)
        print("Generate "+str(self.dred_data)+".")
        dre.Tsne()
        #生成串行文件
        csvp.create_csv_copy(self.dred_data,self.sample_data_output)
        return 0
#===============================5.LDA==========================================        
    def myLDA(self,component_num):
        dre=api.DimensionReduction(components=component_num,
                                   input_data=self.sample_data_output,
                                   dred_data=self.dred_data)
        print("Generate "+str(self.dred_data)+".")
        dre.Lda()
        #生成串行文件
        csvp.create_csv_copy(self.dred_data,self.sample_data_output)
        return 0
    
    
class MLTrainingAndEvaluation(FileNameSetting):
    """5.myAPI中定义得较为完善，且不存在并行问题，因此暂时直接调用myAPI即可"""
    def __init__(self,):
        super().__init__()
    #Deep Neural Network  
    def DNN():
        return 0
    
    
class MLPloting(FileNameSetting):
    """6.画图相关程序，先判断文件是否存在，再进行画图(针对并行做出的优化)"""
    def __init__(self,):
        super().__init__()   
#================================1.draw mean back==============================
    #draw mean back data
    def plot_back_and_all_samples(self,):
        if os.path.isfile(self.back_mean_data)==True:
            dr.draw_data(self.back_mean_data,
                         "Background spectrum",
                         "Channel",
                         "Number of photons",
                           self.folder_3+"Back")
        else:
            print("There is no file named "+str(self.back_mean_data)+".")
#=============================2.draw all cut sample============================            
        #draw all cut samples data        
        if os.path.isfile(self.sample_cut_data)==True:
            dr.draw_data(self.sample_cut_data,
                         "All sample's XAS",
                         "Channel",
                         "Number of photons",
                         self.folder_3+"each_sample_cut",
                         cnum=int(di.global_class_num))
        else:
            print("There is no file named "+str(self.sample_cut_data)+".")
#=========================3.all debacked samples data==========================   
        #draw all debacked samples data     
        if os.path.isfile(self.sample_debacked_data)==True:
            dr.draw_data(self.sample_debacked_data,
                         "All sample's XAS debacked",
                         "Channel",
                         "Number of photons",
                         self.folder_3+"each_sample_debacked",
                         cnum=di.global_class_num)
        else:
            print("There is no file named "+str(self.sample_debacked_data)+".")
        return 0
#==========================4.draw cut mean samples=============================   
    def plot_mean_samples(self,):
        if os.path.isfile(self.sample_cut_mean_data)==True:
            #draw cut mean samples
            dr.draw_data(self.sample_cut_mean_data,
                         "Samples mean XAS",
                         "Channel",
                         "Number of photons",
                         self.folder_3+"sample_cut_mean")
        else:
            print("There is no file named "+str(self.sample_cut_mean_data)+".")
#=======================5.draw debacked mean samples===========================
        if os.path.isfile(self.sample_debacked_mean_data)==True:
            #draw debacked mean samples
            dr.draw_data(self.sample_debacked_mean_data,
                         "Samples mean XAS debacked",
                         "Channel",
                         "Number of photons",
                         self.folder_3+"sample_debacked_mean")
        else:
            print("There is no file named "+str(self.sample_debacked_mean_data)+".")
#========================6.draw scaled mean samples============================        
        if os.path.isfile(self.sample_scaled_mean_data)==True:
            #draw scaled mean samples
            dr.draw_data(self.sample_scaled_mean_data,
                         "Samples mean XAS",
                         "Channel",
                         "Number of photons",
                         self.folder_3+"sample_scaled_mean")
        else:
            print("There is no file named "+str(self.sample_scaled_mean_data)+".")
        return 0
#===============================7.draw PCA=====================================
    def plot_PCA(self,category):
        dr.draw_pca_2D(csv_file=self.dred_data,
                       category_num=category,save_path=self.folder_3+"PCA2D")
        dr.draw_pca_3D(csv_file=self.dred_data,
                       category_num=category,save_path=self.folder_3+"PCA3D")
        dr.draw_pca_scatter(csv_file=self.dred_data,
                            save_path=self.folder_3+"PCAScatter")
        return 0
        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    