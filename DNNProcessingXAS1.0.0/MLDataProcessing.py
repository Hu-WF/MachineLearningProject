#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#------------------------------------------------------------------------------
# Author:      HuWeiFeng
# Created:     2018-04-24
# Finished:    2018-05-06
# E-mail:      674649741@qq.com
# Purpose:     执行数据前处理操作。
#------------------------------------------------------------------------------
import MLFunctions as func

#①②③④⑤⑥⑦为做不同类型实验时，必须要调整的参数
#输入分类数信息,作为全局变量(可供所有.py文件调用)
global_class_num=10#①
#global_class_names=['cancerA','cancerB','normal']#②
global_class_names=['ABS','PES','POM','PPR','PSU','PVC','PVDF','SG','TAR','UV']#②

def main():
#======================1.输入txt，输出load_data.csv=============================
    """建立"file"、"graph"文件夹存储处理后的数据和图表"""
    fo=func.CreateFolder()
    fo.create_folder("file")
    fo.create_folder("graph")
    
    """批量txt转csv，输入txt数据集，最终输出load_data.csv"""
#    ttc=func.TxtToCSV()
    #③
    #方案1-针对每个人的实验都存放一个文件夹的情况(不同文件夹内的相同样本，名称相同)：
#    ttc.txt_to_csv("\\防腐剂实验原始数据集\\1","file\\load_data1.csv")
#    ttc.txt_to_csv("\\防腐剂实验原始数据集\\2","file\\load_data2.csv")
#    ttc.txt_to_csv("\\防腐剂实验原始数据集\\3","file\\load_data3.csv")
#    ttc.special_combine_csv("file\\load_data1.csv","file\\load_data2.csv","file\\load_data3.csv")
    #方案2-针对所有人的实验都放到同一个文件夹的情况(所有相同和不同的样本，名称都不同)：
#    ttc.txt_to_csv("\\生物样本分类数据集","file\\init_load_data.csv")
#    #待重排的文件名；实验人数。
#    ttc.rearrange_csv("file\\init_load_data.csv",3)
    
    #方案3-针对一个人完成所有实验的情况，此时不需要合并文件夹或重排。
#    ttc.txt_to_csv("\\6_binary_class_start100")
    
#    #0值替换成0.01近似，对于样本间差距较大，裁剪后仍有0存在时使用(否则归一化时会报错)。
#    #默认作用于合并或重排后的load_data.csv文件，输出仍为load_data.csv.
#    ttc.replace_sample_zero(0.01)
#    ttc.universal_replace_zero("file\\back_data.csv",0.001)
#    ttc.universal_replace_zero("file\\sample_data.csv",0.001)

#==============2.输入load_data.csv,输出一系列处理后的back.csv、data.csv===========    
    """数据前处理，包括{分割、裁剪、背景均值、去背景、归一化}操作"""
    """输入load_data.csv,输出中间文件备校验，及最终文件data.csv和back.csv"""
    cpp=func.CSVPreProcessing() 
    #分割背景谱和样本谱,起始列到终止列
#    cpp.get_back_data(201,220)#④
#    cpp.get_sample_data(1,200)#⑤
    #裁剪背景谱和样本普部分通道，起始行到终止行
    cpp.cut_back_and_sample_line(50,250)#⑥
    #背景谱求均值，默认对back_cut_data.csv执行
    cpp.back_mean()
    
    #样本普去背景方式1：直接将样本谱对平均值背景去背景
    cpp.deback()
    #样本普去背景方式2：将背景谱分为实验前和试验后两部分，做线性插值产生任意时刻的背景谱，将样本谱一一对应着去背景
    #适用于实验时间跨度较长时（管电流会随时间变化，导致背景谱变化）
    #rotation_sampling=True:表明采用最新的样本自动旋转采样，每个样本都跨整个周期，所以分样本去插值背景
    #cpp.create_interpolation_back(start_back1=1,end_back1=10,start_back2=11,end_back2=20,rotation_sampling=True)
    #cpp.deback(interpolation_back=True)
    
    #样本谱数据正则化，可选标准化还是归一化(scaled_to_01=True),默认执行标准化
#    cpp.scale(scaled_to_01=False)
    cpp.scale(scaled_to_01=True)
    #批量对所有类型的样本谱求均值，以供画图使用
    #{sample_cut_data.csv,sample_debacked_data.csv,sample_scaled_data.csv}
    cpp.all_kinds_sample_mean(global_class_num)
    #输出最终文件data.csv和back.csv
    cpp.finally_data()

#============================3.输出label.csv===================================    
    """根据样本数、分类数自动生成样本标签，标签为one-hot编码格式，输出label.csv"""
    cla=func.CreateLabel(global_class_num)
    #提供独热编码和标签编码两种编码方式可选，采用驼峰命名法
    #'OneHotEncoder'或者default，则采用独热编码；'LabelEncoder'，则采用标签编码
#    cla.encoder('OneHotEncoder')
    cla.encoder('LabelEncoder')

#=====================4.输入data.csv，输出Pca_data.csv==========================
    """光谱数据降维处理，输入data.csv，输出Pca_data.csv作为算法输入""" 
    #输入保留降维后的维数
    dre=func.DReduction(5)#⑦
    #可选多种降维方式，如：PCA、NMF、T-SNE、LDA等，一般采用PCA即可
    dre.Pca()
    #dre.Nmf()
    #dre.Tsne()
    #dre.Lda()
    return 0
    
#主程序入口
if __name__=='__main__':
    main()