
#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     对所有类和函数进行抽象封装。
#==============================================================================
"""
1.基于sklearn、pandas、numpy、scipy、matplotlib等机器学习或数据分析API编写，网址：
	①.http://scikit-learn.org/stable/
	②.http://pandas.pydata.org/
	③.http://www.numpy.org/
	④.https://www.scipy.org/
	⑤.https://matplotlib.org/

2.主要实现以下六大功能：
	①.数据读入整合(生成文件夹，txt批量转csv，合并多个csv，重排单个csv）；
	②.数据前处理（csv分割，裁剪，均值化，归一化，去背景）；
	③.数据降维(PCA降维，可扩展LDA、SVD等其他降维方式)；
	④.训练模型（神经网络训练算法，可扩展SVM、K-neighbor、RF等算法）；
	⑤.模型评估（准确率、多元混淆矩阵、多元评估报告、kappa系数、MCC系数、ROC、AUC）；
	⑥.绘图（背景图X1，光谱图X3，降维图X3）。
	程序可扩展性强，后续可进行大量的横向、纵向扩展。
    
3.2018.6.19更新2.0版：
    ①.重构代码以增强程序可拓展性，进一步抽象封装；
    ②.使data_processing中的子函数变成并行执行（可选）。

4.重构后程序组成：
    ①.mlAPI：所有顶层类和函数封装，不涉及具体过程；
    ②.mlFunctions：次一层类和函数，包括针对具体数据的处理过程；
    ③.data_information：数据包含的原始信息num和names；文件存储名；
    ④.data_processing：从txt到生成back、sample、label四大模块的操作（并行操作）；
    ⑤.training：训练和评估过程；
    ⑥.ploting：根据训练结果作图。
"""
import os
import math
import itertools
#要将整个MLtrainingimport进来，因为若只import两个全局变量，会产生相互引用问题.
#import data_preprocessing
import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#sklearn相关模块
from sklearn import svm
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier#SVM用于多分类时要借助该函数
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation    
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#导入前置信息，后续若要进一步重构代码可考虑从此处入手
import data_information
disi=data_information.SampleInformation()


class FolderProcessing():
    """1.创建、删除文件夹，用于分类存放数据(file)、图表(graph)"""
    def __init__(self):
        self.folder_1="file_output\\"#输出并行file
        self.folder_2="file_temp\\"#输出临时串行file
        self.folder_3="graph_output\\"#输出评估结果图
        self.folder_4="graph_demo\\"#分析时临时绘图
    #自动建立文件夹以存放文件。
    def create_folder(self,path):      
        folder = os.path.exists(path)  
        #判断是否存在文件夹,不存在则创建
        if not folder:           
            os.makedirs(path)        
        else:  
            return 0 
    #批量删除指定文件夹即其内部所有文件(用于初始化)
    def remove_folder(self,*paths): 
        for path in paths:
            filenames=os.listdir(path)
            for filename in filenames:  
                os.remove(path+"\\"+filename)
            os.removedirs(path)
            print("Remove path with files named "+str(path)+'.')
        print("\n")
        return 0

        
class FileProcessing():
    """2.创建空csv文件备用、删除指定名称的csv文件"""
    def __init__(self,):
        pass
    #创建新的空csv文件，若原本存在，则直接覆盖掉
    def create_empty_csv(self,path):
        if os.path.exists(path):
            print("Warning: "+str(path)+" already exist,you are trying to cover this file.")
            os.remove(path)
        data={}
        data=pd.DataFrame(data)
        data.to_csv(path,header=None,index=None)
        return 0
    
    #删除指定名称的文件
    def remove_file(self,file):
        if os.path.exists(file):
            os.remove(file)
        else:
            pass
        return 0
        
        
class CSVGenerating():
    """2.输入：txt files in folder(s)"""
    """  操作：{txt转CSV、CSV合并、CSV重排、0值替换}，保证输出的CSV符合后续要求"""
    """  输出：load_data.csv"""
    def __init__(self):
        pass
#=============================2.1txt转CSV======================================
    #用于将文件夹内批量txt转成单个csv文件，并自动过滤非数字成分。
    #输出文件名可缺省，默认值为load_data.csv。        
    def txt_to_csv(self,txt_url,csv_name):
        #csv_name=self.load_data
        #获取.txt文件夹的路径
        filedir = os.getcwd()+txt_url
        #获取该文件夹中的文件名称列表  
        filenames=os.listdir(filedir)
        #根据列表统计待转换.txt文件个数
        txt_num=len(filenames)
        print("Convert "+str(txt_num)+
              " txts into "+str(csv_name)+" in "+str(txt_url)+".\n")
        #根据文件个数创建相应长度列表，作为列索引名称
        col_list=list(range(1,txt_num+1))
        #row：记录字典的当前行
        #each_txt[]:临时记录当前txt的所有信息
        #all_content_dict={}：key为row，value为当前row对应的each_txt[]
        row=1
        each_txt=[]
        all_content_dict={}      
        #print("Start converting all Txts to CSV......")
        for filename in filenames:
            filepath = filedir+'/'+filename
            #遍历单个txt文件，将每一条line存入each_txt[]中
            for line in open(filepath):
                #过滤txt中非数字成分，strip删除‘\n’分隔符，isdigit判断是否为数字
                #print(line)
                line=line.strip('\n')
                #(若对于CT重建谱，都是小数，没有文本，不能开启过滤，也没必要)
                if line.isdigit()==True:
                    each_txt.append(line)
            #将该条txt[]存入dict{}中
            all_content_dict[row]=each_txt
            row += 1
            #清空当前txt
            each_txt=[]
        #将dict{}存入DataFlame中，表头为col_list
        df=pd.DataFrame(data=all_content_dict,columns=col_list)
        #输出.csv文件，隐藏行列表头
        df.to_csv(csv_name,header=False,index=False)
        #print("Complete conversion.")
        return 0     
#===============================2.2CSV合并===================================== 
    #规则化合并多个.CSV文件(根据光谱数据样本特性，逐列插序合并)。===对应方式1
    #输入多少个文件名，就合并多少个文件。
    def special_combine_csv(self,combined_csv,*csv_names):
        #当前第几个csv
        current_csv=0
        #获取csv文件个数信息
        csv_num=len(csv_names)
        #临时存储所有csv的字典
        stor_dict={}
        for csv_name in csv_names:
            #读入当前csv的数据
            csv=pd.read_csv(csv_name,header=None,index_col=None)
            #获取当前csv的列数信息
            csv_col_num=csv.shape[1]
            #写入的递增列
            col=-1
            while col<csv_col_num-1:
                #单个递增
                col+=1
                #跨单个递增、csv文件数递增的双递增方式写入字典中
                stor_dict[current_csv+csv_num*col]=csv.iloc[:,col]
            #csv文件数递增
            current_csv+=1
        stor_dict=pd.DataFrame(stor_dict)
        #保存成单个csv
        stor_dict.to_csv(combined_csv,header=False,index=False) 
        print("Successfully combine "+str(csv_num)+ " CSV files into "+str(combined_csv)+".\n")
        return 0
#================================2.3CSV重排====================================
    #重排生成的load_data.csv文件。===对应方式2
    #程序思想：假设有3个人做该实验，则先把整个csv按顺序平分成3块（A1，A2，A3），然后按顺序
    #遍历这三块，每次取A1、A2、A3的同一列位置的数据，保存到新的B块中，直到这3块中的所有列
    #都保存到B块中，则B块即为重排后的csv(同一样本的数据都排在一起)。
    #输入待重排的csv文件名；实验人数。
    def rearrange_csv(self,rearrange_csv_name,exp_person_num,rearranged_csv):
        #读入待转换csv文件
        csv=pd.read_csv(rearrange_csv_name,header=None,index_col=None)
        #获取当前csv的列数信息
        csv_col_num=csv.shape[1]
        #per_col_num：每个人做的部分对应的列数=csv总列数/人数
        per_col_num=int(csv_col_num/exp_person_num) 
        #print(per_col_num)
        #记录当前列位置
        col=0   
        #字典暂时存储重排后的csv信息
        arranged={}
        #字典键值
        dict_key=0  
        while col<per_col_num:
            #当前人员（当前块）
            current_person=0
            while current_person<exp_person_num:
                arranged[dict_key]=csv.iloc[:,col+current_person*per_col_num]
                current_person+=1 
                dict_key+=1
            col+=1
        arranged=pd.DataFrame(arranged)
        arranged.to_csv(rearranged_csv,header=False,index=False)
        return 0
#===============================2.4零值替换==================================== 
    #目前暂时用不到
    #通用0值替换程序，给定文件名，替换值    
    def replace_zero(self,file_name,replace_num):
        data=pd.read_csv(file_name,header=None,index_col=None)
        data=data.replace(0,replace_num)
        data.to_csv(file_name,header=False,index=False)
        return 0
    
    
class CSVProcessing():
    """3.输入：load_data.csv"""
    """  操作：{分割、裁剪、均值、去背景、正则化、重命名}5个并行数据前处理操作"""
    """  输出：back.csv、data.csv"""
    def __init__(self,):
        pass    
#========================3.1按列分割出部分csv文件===============================
    #通用的分割程序，可用于任何csv文件分割
    #用于按列分割CSV文件，给定待分割的文件名、起始列、末尾列(从1开始计数)、保存名。    
    def split_csv_by_col(self,file_to_split,start_col,end_col,save_name):
        start_col=int(start_col)
        end_col=int(end_col)
        #由于python是从0开始索引，而思维习惯从1开始数，所以做调整
        start_col=start_col-1
        end_col=end_col
        #读入待处理.csv文件
        data=pd.read_csv(file_to_split,header=None,index_col=None)
        #print(df) 
        #对读入的Dataflame进行按行列位置索引裁剪，并保存（跟MATLAB几乎一样的格式）
        #注意dataflame中的首行首列都是按照0开始索引的，此处跟MATLAB不一样
        data_splited=data.iloc[:,start_col:end_col]
        data_splited.to_csv(save_name,header=False,index=False)
        return 0
#========================3.2按行分割出部分csv文件===============================    
    #通用的按行分割程序，可用于任何csv文件分割，一般用于pca处理后的文件的分割
    #用于按行分割CSV文件，给定待分割的文件名、起始行、末尾行(从1开始计数)、保存名。    
    def split_csv_by_row(self,file_to_split,start_row,end_row,save_name):
        start_row=int(start_row)
        end_row=int(end_row)
        #由于python是从0开始索引，而思维习惯从1开始数，所以做调整
        start_row=start_row-1
        end_row=end_row
        #读入待处理.csv文件
        data=pd.read_csv(file_to_split,header=None,index_col=None)
        #print(df) 
        #对读入的Dataflame进行按行列位置索引裁剪，并保存（跟MATLAB几乎一样的格式）
        #注意dataflame中的首行首列都是按照0开始索引的，此处跟MATLAB不一样
        data_splited=data.iloc[start_row:end_row,:]
        data_splited.to_csv(save_name,header=False,index=False)
        return 0
#=======================3.3按列依顺序合并两个csv文件============================ 
    #默认合并后的A+B.csv中，A文件在前，B在后
    def combine_csv_by_col(self,csv_A,csv_B,csv_AB):
        A=pd.read_csv(csv_A,header=None,index_col=None)
        B=pd.read_csv(csv_B,header=None,index_col=None)
        A_num=A.shape[1]
        B_num=B.shape[1]
        col=0
        while col<B_num:
            #A.iloc[:,A_num+col]=B.iloc[:,col]
            #A[A_num+col]=B.iloc[:,col]
            A[A_num+col]=B[col]
            col+=1
        A=pd.DataFrame(A)
        #print(A)
        A.to_csv(csv_AB,header=False,index=False)
        return 0
    
    
#==============================================================================
#    #合并指定的任意个列集合（列只能以集合的形式指定）
#    def split_and_combine_data(self,input_data,temp_data,output_name,column_sets):
#        fp=FileProcessing()
#        fp.create_empty_csv(temp_data)
#        fp.create_empty_csv(output_name)
#                
#        for column_set in column_sets:
#            self.split_csv_by_col(input_data,column_set[0],column_set[1],temp_data)
#            print(column_sets)
##            print(column_set)
##            print(column_set[0])
#            self.combine_csv_by_col(output_name,temp_data,output_name)  
#        return 0
    
    #从指定的csv文件中挑选出任意列组成新的csv文件（被挑出的类以集合形式给定）(很智能)
    #def split_and_combine_data(self,input_data,output_name,*column_sets):
    def split_and_combine_data(self,input_data,output_name,column_sets):
        #整个运算过程的中间寄存文件，用于辅助运算，最终删除。
        temp_file="temp.csv"
        #读入指定文件
        in_data=pd.read_csv(input_data,header=None,index_col=None)
        #用于判断是否首次读入
        count=0
        for column_set in column_sets: 
            #列序号调整，使得输入从1开始，更直观
            s_col=column_set[0]-1
            e_col=column_set[1]
            #暂存分离出的数据
            temp_data=in_data.iloc[:,s_col:e_col]
            temp_data=pd.DataFrame(temp_data)
            #由于self.combine_csv_by_col()只能基于csv文件执行，所以暂时先写出csv
            temp_data.to_csv(temp_file,header=None,index=None)
            #首个列集合split出来后无法用self.combine_csv_by_col()，所以直接传给output_name
            if count == 0:
                out_data=temp_data
                out_data=pd.DataFrame(out_data)
                out_data.to_csv(output_name,header=None,index=None)
            #非首个，所以启动合并程序
            else:
                self.combine_csv_by_col(output_name,temp_file,output_name)           
                #print(count)
            count+=1
        #删除辅助文件
        fp=FileProcessing()
        fp.remove_file(temp_file)
        return 0
    
#========================3.4CSV另存为指定名称副本===============================    
    #用于创建CSV文件的副本
    def create_csv_copy(self,source_csv,copy_csv):
        data=pd.read_csv(source_csv,header=None,index_col=None)
        data=pd.DataFrame(data)
        data.to_csv(copy_csv,header=None,index=None)
        return 0
#==========================3.5剪切掉指定部分行==================================    
    #用于按行剪切CSV文件，给定起始行、末尾行(从1开始计数)。
    def cut_line(self,start_row,end_row,b_cut_name,a_cut_name):
        start_row=int(start_row)
        end_row=int(end_row)
        start_row=start_row-1
        end_row=end_row-1
        #读入待处理.csv文件
        data=pd.read_csv(b_cut_name,header=None,index_col=None)
        data_cut=data.iloc[start_row:end_row,:]
        #print(dataa)
        data_cut.to_csv(a_cut_name,header=False,index=False) 
        return 0
#===================3.6分类别求均值（适用于样本谱求均值）======================== 
    #对输入sample数据按行、分类别求均值，每类生成一列均值，共category_num列。
    def mean_by_category(self,csv_to_mean,category_num,save_name):
        data=pd.read_csv(csv_to_mean,header=None,index_col=None)
        total_sample_num=data.shape[1]
        per_sample_num=int(total_sample_num/category_num)
        #i：当前循环次数；mean_ed：存储均值化后的数据
        i=0
        mean_ed={}
        while i<category_num:
            mean_ed[i]=data.iloc[:,i*per_sample_num:((i+1)*per_sample_num)-1].mean(1)
            #print(str(i*per_sample_num)+"~"+str(((i+1)*per_sample_num)-1))
            i+=1
        mean_ed=pd.DataFrame(mean_ed) 
        mean_ed.to_csv(save_name,header=False,index=False)
        #print("Generate sample_mean_data.csv.")
        return 0
#======================3.7所有列求均值（适用于back求均值）=======================
    #将所有列分行（通道）求均值（适用于对back求均值）
    def mean_by_line(self,csv_to_mean,save_name):
        data=pd.read_csv(csv_to_mean,header=None,index_col=None)
        data_mean=data.mean(1)
        data_mean.to_csv(save_name,header=None,index=None)
        return 0 
#=============================3.8线性插值算法===================================
    #对两个单列的数组进行线性插值,指定起始、末尾数组，插值点后总点数（包含起始和末尾，所以不能小于2）
    #若想检验该插值算法是否有错，可尝试point_num=2,3,4···，观察插值结果如何
    def linear_interpolation(self,start_csv,end_csv,point_num,save_name):
        #读入起始数组，结尾数组
        data1=pd.read_csv(start_csv,header=None,index_col=None)
        data2=pd.read_csv(end_csv,header=None,index_col=None)
        #data保存插值后的每列
        data=[]
        dic={}
        row=0#当前行位置
        point=0#当前插值点（即当前列）
        #给定一列，先把该列插值结果逐行算完，保存，再算下一列，直到列数等于插值点数
        while point<point_num:
            while row<data1.shape[0]:
                #插值结果=起始值+△值*第几插值列
                info=data1.iloc[row,0]+point*((data2.iloc[row,0]-data1.iloc[row,0])/(point_num-1))
                #print(info)
                data.append(info)
                #print(data)
                row+=1
            #保存插值好的当前一整列
            dic[point]=data
            #print(point)
            #清空data和行位置row信息，为插值下一列做准备
            data=[]
            row=0
            #指到下一列，准备开始下一列插值
            point+=1
        #print(dic)
        dic=pd.DataFrame(dic)
        dic.to_csv(save_name,header=False,index=False)
        return 0             
#===================3.9针对去背景方式的不同，创建插值背景算法=====================
    #创建用于去背景的插值背景。可选择是按时间依次测量还是交叉换样本测量，默认为按时间依次测量
    #注意这里是对cut_back.csv来划分start_back和end_back，所以索引其列位置时要按照cut_back.csv中的位置
    def create_interpolation_back(self,start_back1,end_back1,start_back2,end_back2,rotation_sampling=False):
        #直接针对back_cut_data.csv来划分，省去了划分后还要再去cut的麻烦
        self.split_data(self.back_cut_data,start_back1,end_back1,self.start_back)
        self.split_data(self.back_cut_data,start_back2,end_back2,self.end_back)
        self.mean_by_category(self.start_back,1,self.mean_start_back)
        self.mean_by_category(self.end_back,1,self.mean_end_back)
        #根据样本数，自动计算需插值产生的背景条数
        sample_data=pd.read_csv(self.sample_data,header=None,index_col=None)
        #选择产生插值的方式
        #若采用最新的旋转采样，则每个样本都跨整个采样周期，应分样本去插值背景
        if rotation_sampling:
            interpolation_num=sample_data.shape[1]/disi.class_num
        #若采用传统的手动放置样本，则直接整批样本去插值背景
        else:
            interpolation_num=sample_data.shape[1]
        #开始插值，起始，结尾，插值点数，保存名
        self.linear_interpolation(self.mean_start_back,
                                  self.mean_end_back,interpolation_num,
                                  self.interpolation_back)
        return 0   
#============================3.10去插值背景=====================================
    #样本普去背景。  
    def de_interpolation_back(self,interpolation_back=False):
        #读入待处理.csv文件
        data=pd.read_csv(self.sample_cut_data,header=None,index_col=None)
        #若开启去插值背景的去背景方式，则读入插值背景；否则读入均值背景
        per_sample_num=data.shape[1]/disi.class_num
        if interpolation_back:
            back=pd.read_csv(self.interpolation_back,header=None,index_col=None)
        else:
            back=pd.read_csv(self.back_mean_data,header=None,index_col=None)
        #创建deback的Dataflame
        deback=pd.read_csv(self.sample_cut_data,header=None,index_col=None)
        #行row=j，列column=i
        i=j=0
        back_col=0
        #获取行数data.shape[0]，获取列数data.shape[1]
        while i<data.shape[1]:
            while j<data.shape[0]:
                #启动去插值背景的去背景方式
                if interpolation_back:
                    #考虑到若按样本类来插值去背景，此时插值长度=每类样本数，因此应循环调用插值背景中的列
                    if back_col==per_sample_num:
                        back_col=0
                    #print(back_col)
                    #deback.iloc[j,i]=math.log(back.iloc[j,i]/data.iloc[j,i])
                    #判断back或data中有0值，则log运算会为±无穷，出错，因此直接置零，跳过log
                    if back.iloc[j,back_col]==0 or data.iloc[j,i]==0:
                        deback.iloc[j,i]=0
                    else:
                        deback.iloc[j,i]=math.log(back.iloc[j,back_col]/data.iloc[j,i])
                    j+=1
                    back_col+=1
                #启动去均值背景的去背景方式
                else: 
                    #判断back或data中有0值，则log运算会为±无穷，出错，因此直接置零，跳过log
                    if back.iloc[j,back_col]==0 or data.iloc[j,i]==0:
                        deback.iloc[j,i]=0
                    else:
                        deback.iloc[j,i]=math.log(back.iloc[j,back_col]/data.iloc[j,i])
                    #deback.iloc[j,i]=math.log(back.iloc[j,0]/data.iloc[j,i])
                    j+=1
            j=0
            i+=1
        #print(deback)
        deback.to_csv(self.sample_debacked_data,header=False,index=False)
        if interpolation_back:
            print("Generate sample_debacked_data.csv based on interpolation_back.csv")
        else:
            print("Generate sample_debacked_data.csv based on mean_back.csv")
        return 0
#==============================3.11去均值背景===================================
#    #通用样本普去背景，任意指定样本谱和背景,采用最常用的去均值背景。  
#    def deback(self,sample_data,back_data,debacked_data):
#        #读入待处理.csv文件
#        data=pd.read_csv(sample_data,header=None,index_col=None)
#        back=pd.read_csv(back_data,header=None,index_col=None)
#        #创建deback的Dataflame
#        deback=pd.read_csv(sample_data,header=None,index_col=None)
#        #行row=j，列column=i
#        i=j=0
#        #back_col=0
#        #获取行数data.shape[0]，获取列数data.shape[1]
#        while i<data.shape[1]:
#            while j<data.shape[0]:
#                #启动去均值背景的去背景方式
#                #判断back或data中有0值，则log运算会为±无穷，出错，因此直接置零，跳过log
#                if back.iloc[j,0]==0 or data.iloc[j,i]==0:
#                    deback.iloc[j,i]=0
#                else:
#                    #deback.iloc[j,i]=math.log(back.iloc[j,back_col]/data.iloc[j,i])
#                    deback.iloc[j,i]=math.log(back.iloc[j,0]/data.iloc[j,i])
#                j+=1
#            j=0
#            i+=1
#        #print(deback)
#        deback.to_csv(debacked_data,header=False,index=False)
#        print("Generate debacked_data.csv based on mean_back.csv")
#        return 0
#    
#==============================3.11去均值背景===================================
    #通用样本普去背景，任意指定样本谱和背景,采用最常用的去均值背景。
    #可选择分样本类去各自均值背景，当category_back==False时。
    def deback(self,sample_data,back_data,debacked_data,category_back=False):
        #读入待处理.csv文件
        data=pd.read_csv(sample_data,header=None,index_col=None)
        back=pd.read_csv(back_data,header=None,index_col=None)
        #创建deback的Dataflame
        deback=pd.read_csv(sample_data,header=None,index_col=None)
        #行row=j，列column=i,k用于分样本去各自均值背景时，切换样本背景
        i=j=k=0
        #back_col=0
        #获取行数data.shape[0]，获取列数data.shape[1]
        while i<data.shape[1]:
            #开启分样本去背景
            if category_back == True:
                #k等于当前列位置i除以每块列数，取整，的值
                k=int(i//(data.shape[1]/disi.class_num))
            #否则去综合背景    
            else:
                k=0
            while j<data.shape[0]:      
                #启动去均值背景的去背景方式
#                #判断back或data中有0值，则log运算会为±无穷，出错，因此直接置零，跳过log
#                if back.iloc[j,k]==0 or data.iloc[j,i]==0:
#                    deback.iloc[j,i]=0####此为之前的处理方式，不够严谨(2018-12-04)
                ####新处理方式：背景不会为0；而透射为0时，应置1：
                #判断back或data中有0值，则log运算会为±无穷，出错，因此直接置零，跳过log
                if data.iloc[j,i]==0:
#                    deback.iloc[j,i]=1
                    deback.iloc[j,i]=deback.iloc[j-1,i]
#                    deback.iloc[j,i]=math.log(back.iloc[j,k])
                else:
                    #deback.iloc[j,i]=math.log(back.iloc[j,back_col]/data.iloc[j,i])
                    deback.iloc[j,i]=math.log(back.iloc[j,k]/data.iloc[j,i])
                    #print(k)
                j+=1
            j=0
            i+=1
        #print(deback)
        deback.to_csv(debacked_data,header=False,index=False)
        #print("Generate debacked_data.csv based on mean_back.csv")
        return 0
#=================================3.12正则化===================================     
    #通用：对输入数据进行标准化或者归一化(scale_to_01=True)，默认进行标准化。   
    def data_normalization(self,filename,save_name,scale_to_01=False):
        data=pd.read_csv(filename,header=None,index_col=None)
        #归一化，将数压缩到(0,1)之间：
        if scale_to_01:
            min_max_scaler = preprocessing.MinMaxScaler()
            data_scaled= min_max_scaler.fit_transform(data)
        #标准化，将数按比例缩放到小区间内：
        else:
            data_scaled=preprocessing.scale(data)
        #保存数据
        data_scaled=pd.DataFrame(data_scaled)   
        data_scaled.to_csv(save_name,header=None,index=None)
        #print(data_scaled)
        return 0

    
class CreateLabel():
    """4.根据样本数和分类数自动创建标签文件"""
    #样本数可自动获取，分类数需手动输入
    def __init__(self):
        pass
    #自动获取样本数（无需人工输入）    
    def get_sample_num(self,provide_sample_num_file):
        sample=pd.read_csv(provide_sample_num_file,header=None,index_col=None)
        sample_num=sample.shape[1]
        return sample_num
    #包含独热编码和标签编码两种常用的编码方式（一般来说独热编码比较常用）
    #但是用scikit-learn的confusion_matrix画多元混淆矩阵时，目前只能用标签编码，才不会报错
    #默认采用独热编码方式进行编码，若输入'LabelEncoder'，则采用标签编码    
    def encoder(self,sample_file,category_num,label_csv_name,encoder_mode="OneHotEncoder"):
        #获取样本数信息
        #需加括号，否则认为是调用方法，而非接受其返回值。
        sample_num=self.get_sample_num(provide_sample_num_file=sample_file)
        #创建分类数大小的单位矩阵
        Inum=np.identity(category_num)
        #每类样本的数量num_per_sam
        num_per_sam=int(sample_num/category_num)
        label=[]
        row=col=0
        while row<category_num:
            while col<num_per_sam:
                #编码方式选择独热编码
                if encoder_mode=='OneHotEncoder':
                    label.append(Inum[row,:])
                #编码方式选择标签编码，即0,1,2···
                elif encoder_mode=='LabelEncoder':
                    label.append(row)
                col+=1
            col=0
            row+=1
        label=pd.DataFrame(label).T
        label.to_csv(label_csv_name,header=False,index=False)
        print("Generate label.csv!\n")
        return 0 
    

class DimensionReduction():
    """5.创建用于降维的各种方法,如PCA等，后续可拓展使用LDA等方法"""
    def __init__(self,components,input_data,dred_data):
        self.components=components
        self.data_set_name=input_data
        self.Dred_data=dred_data
#=================================5.1PCA=======================================        
    #PCA降维算法。
    def Pca(self):
        data_set=pd.read_csv(self.data_set_name,header=None,index_col=None)
        data_set=data_set.T   
        #是否开启数据白化，效果对比（可用于写论文分析）
#        pca=PCA(n_components=self.components,whiten=True)
        pca=PCA(n_components=self.components,)
        data_set=pca.fit_transform(data_set)
#        pca.fit(data_set)
#        data_set=pca.transform(data_set)
        #print("Generate Pca_data.csv." )
        print("The interpretability of each component:")
        print(pca.explained_variance_ratio_)
#        print("pca.explained_variance_:")
#        print(pca.explained_variance_)
#        print(pca.components_)
        #计算PCA总和
        sumn=0
        compoments=pca.explained_variance_ratio_
        for comp in compoments:
            sumn+=comp
        print("SUM:")
        print(sumn)
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(self.Dred_data,header=False,index=False)
        return 0
#=================================5.2IPCA======================================    
        #增量PCA降维算法（IPCA）。
    def IPca(self):
        data_set=pd.read_csv(self.data_set_name,header=None,index_col=None)
        data_set=data_set.T   
        ipca=IncrementalPCA(n_components=self.components)
        ipca.fit(data_set)
        data_set=ipca.transform(data_set)
        #print("Generate Pca_data.csv." )
        print("The interpretability of each component:")
        print(ipca.explained_variance_ratio_)
        #计算PCA总和
        sumn=0
        compoments=ipca.explained_variance_ratio_
        for comp in compoments:
            sumn+=comp
        print("SUM:")
        print(sumn)  
        #print(pca.explained_variance_)
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(self.Dred_data,header=False,index=False)
        return 0
#================================5.3NMF========================================    
    #非负矩阵分解降维算法.
    def Nmf(self,):
        data_set=pd.read_csv(self.data_set_name,header=None,index_col=None)
        data_set=data_set.T   
        nmf=NMF(n_components=self.components)
        nmf.fit(data_set)
        data_set=nmf.transform(data_set)
        print("Generate Dre_data.csv." )
        #print("The interpretability of each component:")
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(self.Dred_data,header=False,index=False)
        return 0     
#==============================5.4T-SNE========================================    
    #流形学习降维算法（非线性降维）.
    def Tsne(self,):
        data_set=pd.read_csv(self.data_set_name,header=None,index_col=None)
        data_set=data_set.T   
        tsne=TSNE(n_components=self.components)
        tsne.fit(data_set)
        data_set=tsne.fit_transform(data_set)
        print("Generate Dre_data.csv." )
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(self.Dred_data,header=False,index=False)
        return 0
#==============================5.5.LDA=========================================    
    #线性判别分析降维算法.
    def Lda(self):
        data_set=pd.read_csv(self.data_set_name,header=None,index_col=None)
        data_set=data_set.T   
        lda=LatentDirichletAllocation(n_components=self.components,batch_size=1)
        lda.fit(data_set)
        data_set=lda.fit_transform(data_set)
        print("Generate Dre_data.csv." )
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(self.Dred_data,header=False,index=False)
        return 0
        
      
class MLFrame():
    """6.定义整套抽象机器学习算法框架，包含{splitting，training、evaluation}三大部分"""
    """对于具体机器学习算法，只需继承该类，并重写training部分即可。"""
    def __init__(self,):
        #splitting输入数据集属性
        #由于training和evaluation较完善，因此直接调用，暂时不转移至functions中。
        #后续进一步重构代码时可考虑优化此处
        self.data="file_output\\sample.csv"
        self.label="file_output\\label.csv"
        #training时的中间属性
        self.x_train=self.x_test=self.y_train=self.y_test=[]        
        #通过training输出的属性==evaluation时输入的属性（四大属性）
        #分为三大类：输出精度，基于混淆矩阵的评估、画ROC和AUC
        self.train_accuracy=0#用于输出训练精度
        self.test_accuracy=0#用于输出测试精度
        self.y_predict=[]#用于画混淆矩阵
        self.y_predict_proba=[]#用于画ROC和AUC
        #输入CT重建后的光谱值，该值已经去背景、归一化、降维后，与Dre_data相似，用于预测
        #self.CT_x_data=DataNameSet().CT_x_data
#========================6.1splitting,数据集划分================================   
    #（这里可选是否将标签转为独热编码，因此前面创建标签时一般都创建成标签编码格式）   
    def split_data_set(self,binarize_label=False,transpose_data=True):
        #读取pca降维后的数据集、对应标签
        data=pd.read_csv(self.data,header=None,index_col=None)
        if transpose_data:#190314新增（为使ML和DL完美融合）
            data=data.T
        label=pd.read_csv(self.label,header=None,index_col=None)
        #pca降维后按行存储每条样本，因此label也需对应转置一下
        label=label.T  
        #把pandas的Dataflame(2维)格式转成numpy的array(1维)格式，否则一直提示警告
        #传入的label只需1维格式
        label=np.array(label).ravel()
        #是否需将label转换成one-hot编码格式
        #（画多元ROC时只能用one-hot编码（label_binarize），画多元混淆矩阵只能用标签编码）
        if binarize_label:
            #二值化label,将标签编码转换为独热编码(由于目前sklearn绘制ROC曲线时只能用独热编码)
#            label=label_binarize(label,classes=list(range(0,SampleInformation().class_num)))   
            label=label_binarize(label,classes=list(range(0,disi.class_num)))   
        #数据集分割成训练集和测试集(后续考虑采用k折交叉验证法分割数据集)。
        #测试集占20%，shuffle选择是否打乱顺序（不打乱则效果很差）
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(
                data,label,test_size=0.3,shuffle=True,random_state=2019)
        return 0
#=====================6.2training，训练拟合部分，留空===========================    
    def classifiers(self):
        pass
        return 0
#======================6.3evaluation，评估部分，分四类==========================
#------------------------6.3.1输出训练精度、测试精度-----------------------------
    def output_accuracy(self,):
        print("\n1.训练集和测试集准确率：")
        print("Training accuracy: %f" % self.train_accuracy)
        print("Test accuracy: %f" % self.test_accuracy)
        return 0
#---------------------6.3.2基于混淆矩阵评估，标签编码----------------------------    
    #都是对y_test和y_predict做对比,核心为混淆矩阵。   
    def confusion_matrix_eval(self,category_names):
        #1.多元混淆矩阵
        cm=confusion_matrix(self.y_test,self.y_predict) 
        print("\n2.多元混淆矩阵：")
        print(cm)
        #2.自动评估报告
        #target_names=['class0','class1','class2','class3','class4','class5']
        #统一到DatanameSet()中管理分类样本名称。
        cr=classification_report(self.y_test,self.y_predict,
                                 target_names=category_names)
        print("\n3.自动评估报告：")
        print(cr)
        #3.cohen-kappa系数
        ck=cohen_kappa_score(self.y_test,self.y_predict)
        print("\n4.cohen-kappa系数：")
        print(ck)
        #4.马修斯相关性系数(MCC)
        mcc=matthews_corrcoef(self.y_test,self.y_predict)
        print("\n5.马修斯相关性系数(MCC)：")
        print(mcc)
        return 0
    #绘制彩色的混淆矩阵，正则化可选，default为非正则化
    def draw_confusion_matrix(self, category_names,normalize=False,cmap=plt.cm.Blues):
        cm=confusion_matrix(self.y_test,self.y_predict) 
        classes=category_names
        if normalize:
            #正则化
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
            #print("\n7.Normalized confusion matrix")
        else:
            title='Confusion matrix without normalization'
            print('\n6.Confusion matrix with and without normalization')   
        #画图
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #显示两位小数
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #保存图片
        plt.savefig('graph_output\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph_output\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.figure() 
#-------------------------6.3.3画ROC和AUC曲线，独热编码-------------------------
    #一.绘制多元分类ROC
    def draw_multiple_ROC(self,):#由于
        print('\n7.ROC and AUC curves:')
        #由于sklearn中混淆矩阵和ROC只能分别用标签编码和独热编码，因此目前只能在这里重新拟合计算一遍
        #已通过标签转换直接解决该问题，不必重新训练(190314)
        #self.split_data_set(binarize_label=True)
        #self.classifiers()
        #self.generating_evaluation_parameters()
        y_score=self.y_predict_proba
        #先保留y_test标签编码前的编码，以还原，否则和后续的Cross_validation冲突（2019-03-14）
        y_test_temp=self.y_test
        #转换编码方式--------
        from sklearn.preprocessing import LabelBinarizer
        lb=LabelBinarizer()
        self.y_test=lb.fit_transform(self.y_test)
        y_test=self.y_test
        #print('y_score',y_score.shape,'y_test',y_test.shape)
        #--------
        n_classes=disi.class_num
        classes=disi.class_names
        #Compute ROC curve and ROC area for each class
        fpr=dict()
        tpr=dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw=2
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Plot ROC curves for the multiclass problem
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro({0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro({0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['deepskyblue', 'orange','green', 'brown','darkviolet','sienna','magenta','lightslategrey','y','cyan'])
        #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0}({1:0.2f})'
                     ''.format(classes[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of ROC to multi-class')
        plt.legend(loc="lower right")
        #保存图片
        title='ROC_curve'
        plt.savefig('graph_output\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph_output\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.show()   
        self.y_test=y_test_temp#还原至原始编码方式，供cv使用（2019-03-14）
    #二.绘制二元分类ROC
    def draw_binary_ROC(self,):
        print('\n7.ROC and AUC curves:')
        # 二分类ROC绘制
        plt.figure('ROC')
        y_score=self.y_predict_proba[:,1]
        #print(y_score)
        y_test=self.y_test
        #计算ROC相关参数
        fpr,tpr,_=roc_curve(y_test,y_score)
        #绘制斜线
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='DNN')
        #绘制图片相关信息
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
#        plt.legend(loc='best')
        #保存图片
        title='ROC_curve'
        plt.savefig('graph_output\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph_output\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        return 0
    #三.融合二元分类和多元分类，自动选择绘出ROC        
    def draw_ROC_curves(self,):
        category_num=disi.class_num
        if category_num ==2:
            self.draw_binary_ROC()
        elif category_num > 2:
            self.draw_multiple_ROC()
        else:
            pass
        return 0
#-------------------------6.3.4画PR曲线，独热编码-------------------------------   
    #一.绘制多元分类PR曲线
    def draw_multiple_PR(self,):
        print("\n8.P-R曲线：")
        #计算平均查准率(仅用于2分类情况)
        y_test=self.y_test
        y_score=self.y_predict
        average_precision = average_precision_score(y_test, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))
#        # Plot the Precision-Recall curve
#        #计算PR曲线值
#        precision, recall, _ = precision_recall_curve(y_test, y_score)
#        plt.step(recall, precision, color='b', alpha=0.2,
#                 where='post')
#        plt.fill_between(recall, precision, step='post', alpha=0.2,
#                         color='b')
#        plt.xlabel('Recall')
#        plt.ylabel('Precision')
#        plt.ylim([0.0, 1.05])
#        plt.xlim([0.0, 1.0])
#        ##绘制2分类PR曲线
#        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#                  average_precision))
        #绘制多分类PR曲线：
        #重新传递参数（由于需要独热编码，所以应重新计算，跟前面ROC绘制同样道理）
        self.split_data_set(binarize_label=True)
        self.classifiers()
        y_score=self.y_predict_proba
        Y_test=self.y_test
        n_classes=disi.class_num
        classes=disi.class_names
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        ###绘制多分类的平均查准率
        ### A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                             average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))
        # Plot the micro-averaged Precision-Recall curve
        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                         color='b')  
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
        # Plot Precision-Recall curve for each class and iso-f1 curves
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro({0:0.2f})'
                      ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('{0}({1:0.2f})'
                          ''.format(classes[i], average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        #保存图片
        title='PR_curve'
        plt.savefig('graph_output\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph_output\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.show()  
    #二.绘制二分类PR曲线    
    def draw_binary_PR(self,):
        y_score=self.y_predict_proba[:,1]
        y_test=self.y_test
        #计算P-R
        plt.figure('PRC')
        precision,recall,_=precision_recall_curve(y_test,y_score)
        #计算P-R AUC
        average_precision = average_precision_score(y_test, y_score)
        #绘制PR曲线，填充颜色
        plt.step(recall,precision,color='b',alpha=0.2,where='post')
        plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
        #绘制图片信息
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0,1.05])
        plt.xlim([0.0,1.0])
        plt.title("Precision-Recall curve: AUC={0:0.2f}".format(average_precision))
        #保存图片
        title='PR_curve'
        plt.savefig('graph_output\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph_output\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        return 0
    #三.融合二元分类和多元分类，自动选择绘出PR
    def draw_PR_curves(self,):
        category_num=disi.class_num
        if category_num == 2:
            self.draw_binary_PR()
        else :
            self.draw_multiple_PR()
        return 0
#----------------------------6.3.5交叉验证评估----------------------------------
    #独立定义交叉验证评估方式，需借助classifier参数来评估： 
    #默认折数为5，可自定义，当折数等于样本数时，变成留一法
    def cross_validation(self,k_fold=5,transpose_data=False):
        #读取降维后的数据集、对应标签
        data=pd.read_csv(self.data,header=None,index_col=None)
        label=pd.read_csv(self.label,header=None,index_col=None)
        label=label.T  
        if transpose_data:
            data=data.T#19-03-14新增，解决NN模型问题，使数据按行分布，特征按列分布但是对于降维后的数据使用则会报错,因此默认不开启
#        print(data,label)
        #传入的label只需1维格式
        label=np.array(label).ravel()
        #classifiers()中已返回classifier参数
        clf=self.classifiers()
        scores=cross_val_score(clf,data,label,cv=k_fold)
        print("\n10.K折交叉验证各折准确率:")
        print(scores)
        print("\n11.K折交叉验证平均准确率及置信区间:")
        print("Average accuracy of CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#---------------------------6.3.6对CT重建进行预测-------------------------------
    #主要针对目前提出的论文思路
    def CT_predict(self,):
        ct_x_data=pd.read_csv(self.CT_x_data,header=None,index_col=None)
        clf=self.classifiers()
        ct_predict=clf.predict(ct_x_data)
        num=ct_x_data.shape[0]
        num_list=list(range(0,num))
        print("\n2.CT重建光谱实际标签：")
        print(num_list)
        print("  CT重建光谱预测标签：")
        #改成list格式，输出更好看一些
        ct_predict=list(ct_predict)
        print(ct_predict)
        return 0     
#======6.4综合以上{splitting，training、evaluation}三部分函数，便于直接调用=======
    #也可在具体机器学习算法中重写该部分函数，进行特定的评估
    #该部分函数应通过具体机器学习算法来继承调用，而不能直接调用(因父类classifier为空)
    #训练
    def training(self,transpose_data=False):
        self.split_data_set(binarize_label=False,transpose_data=transpose_data)
        #将分类器训练过程和评估参数生成过程分离，使后续重构更简单

        self.classifiers() 
        return 0
    #评估
    def evaluation(self,K_Fold=10,transpose_data=False,cv=True):
        self.output_accuracy()
        #self.CT_predict()
        self.confusion_matrix_eval(category_names=disi.class_names)
        self.draw_confusion_matrix(category_names=disi.class_names)
        self.draw_confusion_matrix(normalize=True,category_names=disi.class_names)
        self.draw_ROC_curves()
#        self.draw_PR_curves()
        if cv:#默认执行交叉验证评估，CNN模型则关闭CV
            self.cross_validation(k_fold=K_Fold,transpose_data=transpose_data)
        return 0


class NeuralNetwork(MLFrame):
    """7.神经网络算法，继承MLFrame父类，重写ML框架中的classifier()函数即可使用"""
    """不同机器学习算法其实只是分类器classifier不同，因此重写该方法即可"""
    def __init__(self):
        #继承父类的所有属性和方法
        super(NeuralNetwork,self).__init__()
    #重写父类中的classifiers方法    
    def classifiers(self):
        classifier=MLPClassifier(hidden_layer_sizes=(30,30),
                          activation='relu',
                          solver='lbfgs',
                          alpha=0.01,
                          batch_size=1,
                          learning_rate='constant',
                          tol=0.0001,
                          random_state=1)
        classifier.fit(self.x_train,self.y_train)
        #无论用什么机器学习算法训练，都必须输出以下四大参数供评估使用
        self.y_predict=classifier.predict(self.x_test)
        self.y_predict_proba=classifier.predict_proba(self.x_test)  
        self.train_accuracy=classifier.score(self.x_train,self.y_train)
        self.test_accuracy=classifier.score(self.x_test,self.y_test)
        #需返回classifier，供交叉验证评估使用
        return classifier
        
        
class SVM(MLFrame):
    """8.支持向量机算法(Support Vector Machine，SVM)"""
    def __init__(self):
        super(SVM,self).__init__()
    #重写classifiers   
    def classifiers(self):
        #对于多元分类，用SVM，且还要画ROC曲线的，必须使用OneVsRestClassifier，否则能分类但无法画出ROC
        classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,))
        classifier.fit(self.x_train,self.y_train)
        #输出四大参数供评估使用
        self.y_predict=classifier.predict(self.x_test)
        #这里用predict_proba和decision_function都能画出ROC曲线，且不同，原因？
        #self.y_predict_proba=classifier.predict_proba(self.x_test)  
        self.y_predict_proba=classifier.decision_function(self.x_test)
        self.train_accuracy=classifier.score(self.x_train,self.y_train)
        self.test_accuracy=classifier.score(self.x_test,self.y_test)
        #返回classifier，供交叉验证评估使用
        return classifier


class KNeighbors(MLFrame):
    """9.K近邻算法"""
    def __init__(self,):
        super(KNeighbors,self).__init__()
    #重写classifiers
    def classifiers(self):
        classifier=KNeighborsClassifier(n_neighbors=1,
                                        weights='uniform',
                                        algorithm='auto',
                                        leaf_size=6,p=2,
                                        metric='minkowski',
                                        metric_params=None,n_jobs=1)

        classifier.fit(self.x_train,self.y_train)
        #输出四大参数供评估使用
        self.y_predict=classifier.predict(self.x_test)
        #这里用predict_proba和decision_function都能画出ROC曲线，且不同，原因？
        self.y_predict_proba=classifier.predict_proba(self.x_test)  
#        self.y_predict_proba=classifier.decision_function(self.x_test)
        self.train_accuracy=classifier.score(self.x_train,self.y_train)
        self.test_accuracy=classifier.score(self.x_test,self.y_test)
        #返回classifier，供交叉验证评估使用
        return classifier

    
class Drawing():
    """10.定义绘图相关属性和方法，包含{背景图X1、光谱图X3、PCA图X3}三部分，共7张"""
    #统一定义图形相关参数，统一命名图片存储名称
    def __init__(self,):
        super().__init__()
        self.colors=['b','c','g','k','m','r','y','w','b','c']
        self.markers=['o','v',',','*','x','s']
#        self.markers=['o','v',]
#        self.line_type=['b--','g-',]
        self.line_type=['k','k-','k--','k:','k','k-',]
        self.sample_names=disi.class_names
        #self.class_num=SampleInformation().class_num
        self.back_name=('Background spectrum',)#逗号不能少   
#=================10.1定义通用绘图函数，用于绘制背景图和光谱图====================     
    def draw_data(self,csv_name,title,xlabel="Channel",ylabel="Number of photons",save_name="test",cnum=0):
        
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        plt.figure(save_name)
        #读入csv数据
        data=pd.read_csv(csv_name,header=None,index_col=None)
        i=0
        x = np.arange(data.shape[0]) 
        #print(data.shape[1])
        if data.shape[1]==1:
            #说明是back_data
            y=data.iloc[:,0]
#            plt.plot(x,y,self.line_type,marker=self.markers[i],linewidth=1,) 
#            plt.plot(x,y,self.line_type,c=self.colors[i],marker=self.markers[i],linewidth=1,) 
            #plt.plot(x,y,self.line_type,c=self.colors[i],linewidth=1,) 
            plt.plot(x,y,linewidth=1,) 
            plt.legend(self.back_name)
        else:
            #说明是sample_data
            while i<data.shape[1]:#获取列数
                y=data.iloc[:,i]
                if cnum==0:
                    #说明是平均值后的sample_data，则直接画
#                    plt.plot(x,y,self.line_type,linewidth=1,) 
                    #plt.plot(x,y,self.line_type,c=self.colors[i],marker=self.markers[i],linewidth=1,) 
                    #plt.plot(x,y,self.line_type,c=self.colors[i],linewidth=1,) 
#                    plt.plot(x,y,self.line_type[i],linewidth=1,) 
#                    plt.plot(x,y,self.line_type[i],linewidth=1,c='black') 
                    
                    plt.plot(x,y,linewidth=1,) 
                    i+=1
                
                else:
                    #否则是画所有的sample_data，同一类的画同一种颜色
                    per_num=data.shape[1]/cnum
                    j=int(i/per_num)
                    #print(j)
                    plt.plot(x,y,c=self.colors[j],linewidth=1,)
#                    plt.plot(x,y,self.line_type[j],linewidth=1,)
#                    plt.plot(x,y,self.markers[j],linewidth=1,)
                    i+=1
#            plt.legend(self.sample_names)
            if cnum==0:
                #只有平均值类的数据需要标出标注          
                plt.legend(self.sample_names)
            #plt.legend(self.sample_names)
        #设置图片外围信息
        plt.title(title,fontsize=10)
        plt.xlabel(xlabel,fontsize=10)
        plt.ylabel(ylabel,fontsize=10)
        #设置刻度标记大小labelsize
        plt.tick_params(axis='both',which='major',labelsize=10)
        #画右上角标注,注意要将标签信息都放在同一个括号内才可以(建立元组储存)。
        #自动保存图片(pdf格式比png高清很多)
        plt.savefig(save_name+'.pdf',bbox_inches='tight',)
        plt.savefig(save_name+'.png',bbox_inches='tight',dpi=128)
        plt.show()
        return 0
#======================10.2定义调试时的绘图工具，可画指定列=======================    
    #该函数用于MLDemo中，用于分析数据时，灵活画图观察使用。
    def draw_specific_col(self,csv_name,save_name="test",title='title',*cols):
        plt.figure(save_name)
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#        title=title
#        xlabel="Channel"
        xlabel="Tube voltage(keV)"
        ylabel="X-ray absorption coefficient"
        #读入csv数据
        data=pd.read_csv(csv_name,header=None,index_col=None)
        x = np.arange(data.shape[0]) 
        #print(x)
        """换算通道数和能量"""
        x=0.2007639*(x-0.3561)#标定换算公式，可选择注销
        #print(x,x.shape)
        #print(data.shape[1]) 
        #对应指定的所有列，遍历画出全部
        for col in cols:
            y=data.iloc[:,col-1]
            plt.plot(x,y,linewidth=1,) 
        plt.legend(self.sample_names)
        #plt.legend(self.sample_names)
        #设置图片外围信息
        plt.title(title,fontsize=10)
        plt.xlabel(xlabel,fontsize=10)
        plt.ylabel(ylabel,fontsize=10)
        #设置刻度标记大小labelsize
        plt.tick_params(axis='both',which='major',labelsize=10)
        #画右上角标注,注意要将标签信息都放在同一个括号内才可以(建立元组储存)。
        #自动保存图片(pdf格式比png高清很多)
        plt.savefig('graph_demo\\'+save_name+'.pdf',bbox_inches='tight',)
        plt.savefig('graph_demo\\'+save_name+'.png',bbox_inches='tight',dpi=128)
        plt.show()
        return 0   
#============================10.3绘制3种PCA图==================================
    #取前2主成分画二维图    
    def draw_pca_2D(self,csv_file,category_num,save_path):
        plt.figure("2D")
        data=pd.read_csv(csv_file,header=None,index_col=None)
        #each_num：每类样本个数
        each_num=int(data.shape[0]/category_num)
        x, y = data[0], data[1]
        i=0
        while i<category_num:
            s_row=i*each_num
            e_row=((i+1)*each_num)-1
#            plt.scatter(x[s_row:e_row], y[s_row:e_row],marker=self.markers[i],c='black')
#            plt.scatter(x[s_row:e_row], y[s_row:e_row], c=self.colors[i],marker=self.markers[i])
            plt.scatter(x[s_row:e_row], y[s_row:e_row],)
            i+=1
        #plt.title('PCA',fontsize=16)
#        ax.set_zlabel('PCA1',fontsize=6) 
        plt.xlabel('PCA1',fontsize=12)
        plt.ylabel('PCA2',fontsize=12)
#        ax.set_ylabel('PCA2',fontsize=6)
#        ax.set_xlabel('PCA1',fontsize=6)
        #设置刻度标记大小labelsize
        plt.tick_params(axis='both',which='major',labelsize=10)
        #自动保存图片
        plt.legend(self.sample_names)
        plt.savefig(save_path+'.pdf',bbox_inches='tight',)
        plt.savefig(save_path+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        return 0   
    #取前3主成分画三维图    
    def draw_pca_3D(self,csv_file,category_num,save_path):
        plt.figure("3D")
        data=pd.read_csv(csv_file,header=None,index_col=None)
        #each_num：每类样本个数
        each_num=int(data.shape[0]/category_num)
        x, y, z = data[0], data[1], data[2]
        # 创建一个三维的绘图工程
        ax = plt.subplot(111, projection='3d')
        i=0
        while i<category_num:
            s_row=i*each_num
            e_row=((i+1)*each_num)-1
            #ax.scatter(x[s_row:e_row], y[s_row:e_row], z[s_row:e_row], c=self.colors[i],marker=self.markers[i]) 
            ax.scatter(x[s_row:e_row], y[s_row:e_row], z[s_row:e_row],) 
            i+=1
        #plt.title('PCA',fontsize=16)
        ax.set_zlabel('PCA1',fontsize=12) 
        ax.set_ylabel('PCA2',fontsize=12)
        ax.set_xlabel('PCA3',fontsize=12)
        #设置刻度标记大小labelsize
        plt.tick_params(axis='both',which='major',labelsize=10)
        #自动保存图片
        plt.legend(self.sample_names)
        plt.savefig(save_path+'.pdf',bbox_inches='tight',)
        plt.savefig(save_path+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        return 0    
    #取前4主成分画散点图
    def draw_pca_scatter(self,csv_file,save_path):
        plt.figure("4D")
        data=pd.read_csv(csv_file,header=None,index_col=None)
#        title=["PCA1[46.27%]","PCA2[18.24%]","PCA3[9.45%]","PCA4[5.6%]"]
        title=["PCA1","PCA2","PCA3","PCA4"]
        i=0
        x = np.arange(data.shape[0]) 
        while i<4:
            plt.subplot(221+i)
            plt.scatter(x,data.iloc[:,i],s=1)
            plt.title(title[i])
            i+=1
            plt.grid(True)
            #此处可修改PCA坐标轴范围，可根据作图效果需要手动设置，默认不开启
            #plt.xlim((0,200))
            plt.ylim((-10,10))
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, 
                            hspace=0.35,wspace=0.25)
        #plt.legend(self.sample_names)
        plt.savefig(save_path+'.pdf',bbox_inches='tight',)
        plt.savefig(save_path+'.png',bbox_inches='tight',dpi=256)
        plt.show()        
        return 0  
    

            
    