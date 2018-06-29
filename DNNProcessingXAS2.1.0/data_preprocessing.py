#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-19
# Version:     2.0.0
# E-mail:      674649741@qq.com
# Purpose:     {建文件夹；txt转csv；分割；裁剪；均值；去背景；归一化；降维；生成标签}
#==============================================================================
import mlAPI as api
import mlFunctions as f
import data_information as di
disi=di.SampleInformation()

def main():  
#====================1.删除指定文件夹及内部文件(用于初始化)=======================
    fp=api.FolderProcessing()
    fp.remove_folder("file_temp","file_output","graph_output","graph_demo")
#=============================2.生成文件夹======================================
    fp.create_folder("file_temp")
    fp.create_folder("file_output")
    fp.create_folder("graph_output")
    fp.create_folder("graph_demo")
#==============================3.txt转csv======================================
    ttc=f.MLTxt2Csv()
#    ttc.ml_txt_to_csv("\\6")
    ttc.ml_txt_to_csv("\\1小鼠肝脏实验_删除两组容器背景\\4")
#==============4.{分割；裁剪；均值；去背景；归一化}（可选操作）===================
    cp=f.MLCsvProcessing()
    cp.get_back_data(1,20)
    cp.get_sample_data(21,220)
    cp.cut_back_and_sample_line(3,230)
    cp.mean_back()
    cp.deback_one_back()
    #cp.scale(scaled_to_01=True)
    #cp.scale(scaled_to_01=False)
    cp.mean_all_kinds_sample(disi.class_num)
#=============================5.生成标签========================================
    cl=f.MLCreateLabel()
    cl.create_label(category=disi.class_num,encode_mode='LabelEncoder')
    #cl.create_label(category=global_class_num,encode_mode='OneHotEncoder')
#=============================6.降维操作========================================    
    dre=f.MLDRe()
    dre.myPCA(6)
    return 0

if __name__=='__main__':
    main()

