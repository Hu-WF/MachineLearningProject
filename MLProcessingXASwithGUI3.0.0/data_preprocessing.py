#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     {建文件夹；txt转csv；分割；裁剪；均值；去背景；归一化；降维；生成标签}
#==============================================================================
#import sys
#sys.setrecursionlimit(1000000000)
import mlAPI as api
import mlFunctions as f
import data_information as di
disi=di.SampleInformation()

#================================GUI参数绑定====================================
#定义传入参数
#txt_dir="\\原始数据20170417"
txt_dir="\\Null"
#back数据
back_data_s='n'
back_data_e='n'
back_data=[]
#sample数据
sample_data_s='n'
sample_data_e='n'
sample_data=[]
#通道数据
channel_s='n'
channel_e='n'
#判断裁剪，均值，去背景，归一化，降维，生成标签是否执行,1代表执行
botton_state=[1,1,1,1,1,1]
#传入下拉框状态
com_1_back_state='null'
com_2_scale_state='null'
com_3_dre_state='null'
com_4_model_state='null'
#保留降维维度数据
dred_num=0

#==============================================================================

def main():  
#====================1.删除指定文件夹及内部文件(用于初始化)=======================
    fp=api.FolderProcessing()
#    fp.remove_folder("file_temp","file_output","graph_output","graph_demo")
#=============================2.生成文件夹======================================
    fp.create_folder("file_temp")
    fp.create_folder("file_output")
    fp.create_folder("graph_output")
    fp.create_folder("graph_demo")
#==============================3.txt转csv======================================
    ttc=f.MLTxt2Csv()
#    ttc.ml_txt_to_csv("\\6")
#    ttc.ml_txt_to_csv("\\1小鼠肝脏实验_原始数据\\4")
#    ttc.ml_txt_to_csv("\\1小鼠肝脏实验_删除两组容器背景\\4")
#    ttc.ml_txt_to_csv("\\原始数据20170417")
    ttc.ml_txt_to_csv(txt_dir)
#==============4.{分割；裁剪；均值；去背景；归一化}（可选操作）===================
    cp=f.MLCsvProcessing()
#----4.1分割-------------------------------------------------------------------
    #4.1.1普通分割方法
#    cp.get_back_data(1,10)
#    cp.get_sample_data(11,460)

        
#    cp.get_back_data(back_data_s,back_data_e)
#    cp.get_sample_data(sample_data_s,sample_data_e)
    #4.1.2更智能的get back and sample方法（以任意数量的列集合给出，即可自动生成）
    #将列表back_data拆开，将其元素作为输入
    cp.get_back_data_wisely(*back_data)
    cp.get_sample_data_wisely(*sample_data)
#    cp.get_back_data_wisely([21,30],[131,140])
#    cp.get_sample_data_wisely([31,130],[141,240])
#----裁剪----------------------------------------------------------------------
    if botton_state[0] == 1:
        cp.cut_back_and_sample_line(channel_s,channel_e)
#----4.2背景均值---------------------------------------------------------------
    if botton_state[1] == 1:
        if com_1_back_state == "总平均背景":
            #4.2.1求统一背景均值
            cp.mean_one_back() 
        elif com_1_back_state == "分样本背景":
        #4.2.2分样本类求背景均值
            cp.mean_category_back()
#----4.3去背景-----------------------------------------------------------------
    if botton_state[2] == 1:
        if com_1_back_state == "总平均背景":
            #4.3.1去单个背景
            cp.deback_one_back()
        elif com_1_back_state == "分样本背景":
            #4.3.2分样本类去背景
            cp.deback_category_back()
#----4.4归一化-----------------------------------------------------------------
    #4.4.1：0-1归一化
    if botton_state[3] == 1:
        if com_2_scale_state == "归一化":
            cp.scale(scaled_to_01=True)
        elif com_2_scale_state == "标准化":
            #4.4.2：标准化
            cp.scale(scaled_to_01=False)
#----4.5样本均值---------------------------------------------------------------
    cp.mean_all_kinds_sample(int(di.global_class_num))
#    print(di.global_class_num)
#    cp.mean_all_kinds_sample(di.global_class_num)
#=============================5.生成标签========================================
    if botton_state[4] == 1:
        cl=f.MLCreateLabel()
        cl.create_label(di.global_class_num,encode_mode='LabelEncoder')
    #cl.create_label(category=global_class_num,encode_mode='OneHotEncoder')
#=============================6.降维操作======================================== 
    if botton_state[5] == 1:
        dre=f.MLDRe()
        global pca_ret
        if com_3_dre_state == 'PCA':
            pca_ret=list(dre.myPCA(dred_num))
        elif com_3_dre_state == 'IPCA':
            pca_ret=list(dre.myIPCA(dred_num))
        elif com_3_dre_state == 'LDA':
            pca_ret=list(dre.myLDA(dred_num))
        elif com_3_dre_state == 'NMF':
            pca_ret=list(dre.myNMF(dred_num))
        elif com_3_dre_state == 'T-SNE':
            pca_ret=list(dre.myTSNE(dred_num))
    return 0

if __name__=='__main__':
    main()

