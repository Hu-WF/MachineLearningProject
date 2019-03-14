#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
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

#    ttc.ml_txt_to_csv("\\5仙草冻实验数据\\1")
#    ttc.ml_txt_to_csv("\\6小鼠肝硬化实验181210\\3")
#    ttc.ml_txt_to_csv("\\7MouseLiver181210")

    ttc.ml_txt_to_csv("\\3D打印耗材分类实验")
#    ttc.ml_txt_to_csv("\\sl_rebuilt_XAS\\")
#==============4.{分割；裁剪；均值；去背景；归一化}（可选操作）===================
    cp=f.MLCsvProcessing()
#----4.1分割-------------------------------------------------------------------
    #4.1.1普通分割方法
#    cp.get_back_data(1,10)
#    cp.get_sample_data(11,910)
#    cp.get_sample_data(11,210)
    #4.1.2更智能的get back and sample方法（以任意数量的列集合给出，即可自动生成）
    cp.get_back_data_wisely([1,10])
#    cp.get_sample_data_wisely([11,110],[211,310],[311,410],[411,510],[611,710],[811,910])#['ePA-CF',  'PLA', 'ABS', 'TPV', 'PVA', 'UV9400']
    cp.get_sample_data_wisely([11,110],[211,310],[311,410],[711,810],[811,910])#['ePA-CF',  'PLA', 'ABS',  'TPE', 'UV9400']
    
#    cp.get_sample_data_wisely([11,110],[211,310],[311,410],[811,910])#['ePA-CF',  'PLA', 'ABS', 'UV9400']
    

#----裁剪----------------------------------------------------------------------
#    cp.cut_back_and_sample_line(7,120)
#    cp.cut_back_and_sample_line(7,160)
#    cp.cut_back_and_sample_line(7,200)
    cp.cut_back_and_sample_line(10,250)
    
#    cp.cut_back_and_sample_line(7,130)
#    cp.cut_back_and_sample_line(10,160)
#    cp.cut_back_and_sample_line(10,200)
#    cp.cut_back_and_sample_line(7,250)
#----4.2背景均值---------------------------------------------------------------
    #4.2.1求统一背景均值
    cp.mean_one_back() 
    #4.2.2分样本类求背景均值
#    cp.mean_category_back()
#----4.3去背景-----------------------------------------------------------------
    #4.3.1去单个背景
    cp.deback_one_back()
    #4.3.2分样本类去背景
#    cp.deback_category_back()
#----4.4归一化-----------------------------------------------------------------
    #4.4.1：0-1归一化
    cp.scale(scaled_to_01=True)
    #4.4.2：标准化
#    cp.scale(scaled_to_01=False)
#----4.5样本均值---------------------------------------------------------------
    cp.mean_all_kinds_sample(disi.class_num)
#=============================5.生成标签========================================
    cl=f.MLCreateLabel()
    cl.create_label(category=disi.class_num,encode_mode='LabelEncoder')
#    cl.create_label(category=global_class_num,encode_mode='OneHotEncoder')
#=============================6.降维操作======================================== 
    #使用CNN模型、SAE-DNN模型时不能用PCA降维（CNN已有特征提取功能）
#    dre=f.MLDRe()
#    dre.myPCA(10)
    return 0

if __name__=='__main__':
    main()

