#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#------------------------------------------------------------------------------
# Author:      HuWeiFeng
# Created:     2018-04-24
# Finished:    2018-05-06
# E-mail:      674649741@qq.com
# Purpose:     包含{数据前处理、ML训练和评估算法、绘图}三个部分的所有类和函数。
#------------------------------------------------------------------------------
"""
1.基于sklearn、pandas、numpy、scipy、matplotlib等机器学习或数据分析API编写，网址：
	①.http://scikit-learn.org/stable/
	②.http://pandas.pydata.org/
	③.http://www.numpy.org/
	④.https://www.scipy.org/
	⑤.https://matplotlib.org/

2.项目包含以下四个模块：
	①.MLFunctions.py：程序的所有类和函数存放于此处，集中管理和调用；
	②.MLDataProcessing.py：针对不同类型的实验，选择性调用数据处理程序；
	③.MLTraining.py：控制模型进行训练和评估；
	④.MLPloting.py：绘制数据图表。
	运行次序为①②③④或①②④③。

2.主要实现以下六大功能：
	①.数据读入整合(生成文件夹，txt批量转csv，合并多个csv，重排单个csv）；
	②.数据前处理（csv分割，裁剪，均值化，归一化，去背景）；
	③.数据降维(PCA降维，可扩展LDA、SVD等其他降维方式)；
	④.训练模型（神经网络训练算法，可扩展SVM、K-neighbor、RF等算法）；
	⑤.模型评估（准确率、多元混淆矩阵、多元评估报告、kappa系数、MCC系数、ROC、AUC）；
	⑥.绘图（背景图X1，光谱图X3，降维图X3）。
	程序可扩展性强，后续可进行大量的横向、纵向扩展。
"""

import os
import math
import itertools
#要将整个MLtrainingimport进来，因为若只import两个全局变量，会产生相互引用问题.
import MLDataProcessing

import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle 
import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier#SVM用于多分类时要借助该函数

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation    
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
#from sklearn.multiclass import OneVsRestClassifier


class DataNameSet():
    """1.包含数据处理过程中所有需存储的.csv文件名，便于集中管理"""
    def __init__(self):
        #需建立名为“file”的文件夹来存放以下.CSV文件。
        #load_data.csv:      txt转成单个csv时存放的位置;
        
        #back_data.csv:      原始背景谱；
        #back_cut_data.csv:  裁剪掉部分通道后的背景谱；
        #back_mean_data.csv: 求均值后的背景谱；
        #back.csv:           经过以上处理后，最终可用的背景谱。        

        #sample_data.csv:          原始样本谱；
        #sample_cut_data.csv:      裁剪掉部分通道后的样本谱；
        #sample_mean_data.csv:     分类求均值的样本普(仅供画图使用)；
        #sample_debacked_data.csv: 去背景后的样本普；
        #sample_scaled_data.csv:   正则化后的样本普；
        #Dre_data.csv:             降维后的样本普；
        #data.csv:                 经过以上处理后，最终可用的样本普。
        
        #label.csv:                自动生成的样本标签
        self.load_data="file\\load_data.csv"
        
        self.back_data="file\\back_data.csv"
        self.sample_data="file\\sample_data.csv"
        
        self.back_cut_data="file\\back_cut_data.csv"
        self.sample_cut_data="file\\sample_cut_data.csv"
        
        self.back_mean_data="file\\back_mean_data.csv"
        #下面这三个均值仅供画图使用，不参与余下计算过程
        self.sample_mean_data="file\\sample_mean_data.csv"
        self.sample_debacked_mean_data="file\\sample_debacked_mean_data.csv"
        self.sample_scaled_mean_data="file\\sample_scaled_mean_data.csv"
        
        self.sample_debacked_data="file\\sample_debacked_data.csv" 
        self.sample_scaled_data="file\\sample_scaled_data.csv"
        
        self.finally_back_data="file\\back.csv"
        self.finally_sample_data="file\\data.csv"
        
        self.Dre_data="file\\Dre_data.csv"  
        self.label_data="file\\label.csv"  
        #用于采用插值背景，去背景时的文件名
        self.start_back="file\\start_back.csv"
        self.end_back="file\\end_back.csv"
        self.mean_start_back="file\\mean_start_back.csv"
        self.mean_end_back="file\\mean_end_back.csv"
        self.interpolation_back="file\\interpolation_back.csv"
        #用于对CT重建结果进行预测
        self.CT_x_data="file_ct\\ct_x_data.csv"

class SampleInformation():
    """2.包含数据分析评估过程需要的信息"""
    def __init__(self,):
        #需外部传入的参数，从MLDataProcessing中传入全局变量（样本名和分类数）
        self.class_num=MLDataProcessing.global_class_num
        self.class_names=MLDataProcessing.global_class_names
        #自动计算得到的参数
        self.sample_num_array=list(range(0,MLDataProcessing.global_class_num-1))
        


class CreateFolder():
    """3.创建文件夹，用于分类存放数据(file)、图表(graph)"""
    def __init__(self):
        self.data_folder="file"
        self.graph_folder="graph"

    #自动建立文件夹以存放文件。
    def create_folder(self,path):      
        folder = os.path.exists(path)  
        #判断是否存在文件夹,不存在则创建
        if not folder:           
            os.makedirs(path)        
        else:  
            return 0
        
        
class TxtToCSV():
    """4.输入：txt files in folder(s)"""
    """  操作：{txt转CSV、CSV合并、CSV重排、0值替换}，保证输出的CSV符合后续要求"""
    """  输出：load_data.csv"""
    def __init__(self):
        self.output_csv=DataNameSet().load_data

#---------------------------4.1txt转CSV---------------------------------------- 
    #用于将文件夹内批量txt转成单个csv文件，并自动过滤非数字成分。
    #输出文件名可缺省，默认值为load_data.csv。        
    def txt_to_csv(self,txt_url,csv_name=DataNameSet().load_data):
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
        #print("Generate load_data.csv.\n")
        return 0
        
#------------------------------4.2CSV合并--------------------------------------   
    #规则化合并多个.CSV文件(根据光谱数据样本特性，逐列插序合并)。===对应方式1
    #输入多少个文件名，就合并多少个文件。
    def special_combine_csv(self,*csv_names):
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
        stor_dict.to_csv(DataNameSet().load_data,header=False,index=False) 
        print("Successfully combine "+str(csv_num)+ " CSV files into load_data.csv.\n")
        return 0
    
#-------------------------------4.3CSV重排-------------------------------------    
    #重排生成的load_data.csv文件。===对应方式2
    #程序思想：假设有3个人做该实验，则先把整个csv按顺序平分成3块（A1，A2，A3），然后按顺序
    #遍历这三块，每次取A1、A2、A3的同一列位置的数据，保存到新的B块中，直到这3块中的所有列
    #都保存到B块中，则B块即为重排后的csv(同一样本的数据都排在一起)。
    #输入待重排的csv文件名；实验人数。
    def rearrange_csv(self,rearrange_csv_name,exp_person_num):
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
        arranged.to_csv(self.output_csv,header=False,index=False)
        return 0

#------------------------------4.4零值替换-------------------------------------    
    #0值替换成0.01近似，对于样本间差距较大，裁剪后仍有0存在时使用(否则归一化时会报错)。
    #默认作用于合并、重排后的load_data.csv文件，输出仍为load_data.csv.
#    def replace_zero(self,replace_num):
#        data=pd.read_csv(self.output_csv,header=None,index_col=None)
#        data=data.replace(0,replace_num)
#        data.to_csv(self.output_csv,header=False,index=False)
        
    #通用0值替换程序，给定文件名，替换值    
    def universal_replace_zero(self,file_name,replace_num):
        data=pd.read_csv(file_name,header=None,index_col=None)
        data=data.replace(0,replace_num)
        data.to_csv(file_name,header=False,index=False)
        
    #默认对样本谱进行0值替换
    def replace_sample_zero(self,num):
        self.universal_replace_zero(self.output_csv,num)
        return 0
    
#--------------------4.5按顺序合并两个csv文件中的每列文件------------------------ 
    #默认合并后的A+B.csv中，A文件在前，B在后
    def combine_csv(self,csv_A,csv_B,csv_AB):
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
        
    
    
class CSVPreProcessing(DataNameSet):#继承DataNameSet()类
    """5.输入：load_data.csv"""
    """  操作：{分割、裁剪、均值、去背景、正则化、重命名}5个数据前处理操作"""
    """  输出：back.csv、data.csv"""
    def __init__(self,):
        super().__init__()
        
#-----------------------------5.1分割------------------------------------------
    #通用的分割程序，可用于任何csv文件分割
    #用于按列分割CSV文件，给定待分割的文件名、起始列、末尾列(从1开始计数)、保存名。    
    def split_data(self,file_to_split,start_col,end_col,save_name):
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
    
    #通用的按列分割程序，可用于任何csv文件分割，一般用于pca处理后的文件的分割
    #用于按行分割CSV文件，给定待分割的文件名、起始行、末尾行(从1开始计数)、保存名。    
    def split_data_by_row(self,file_to_split,start_row,end_row,save_name):
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
    
    #借助上面的分割函数生成背景谱。默认数据源为load_data.csv.
    def get_back_data(self,s_col,e_col):
        self.split_data(self.load_data,s_col,e_col,self.back_data)
        print("Generate back_data.csv.")
        return 0

    #借助分割函数生成样本普。默认数据源为load_data.csv.
    def get_sample_data(self,s_col,e_col):
        self.split_data(self.load_data,s_col,e_col,self.sample_data)
        print("Generate sample_data.csv.")
        return 0
    
#-------------------------------5.2剪切----------------------------------------    
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
    
#    #借助以上剪切函数剪切背景谱。
#    def back_cutting_data(self,s_row,e_row):
#        self.cut_line(s_row,e_row,self.back_data,self.back_cut_data)
#        print("Generate back_cut_data.csv.")        
#        return 0
#   
#    #借助剪切函数剪切样本谱。     
#    def sample_cutting_data(self,s_row,e_row):
#        self.cut_line(s_row,e_row,self.sample_data,self.sample_cut_data)
#        print("Generate sample_cut_data.csv.")
#        return 0
    
    #借助剪切函数同时剪切背景谱和样本谱。     
    def cut_back_and_sample_line(self,s_row,e_row):
        #裁剪背景谱
        self.cut_line(s_row,e_row,self.back_data,self.back_cut_data)
        print("Generate back_cut_data.csv.")
        #裁剪样本谱
        self.cut_line(s_row,e_row,self.sample_data,self.sample_cut_data)
        print("Generate sample_cut_data.csv.")
        return 0

#---------------------------------5.3均值--------------------------------------
#    #通用求均值：对输入的to_mean_file按行求均值，生成一列,储存名为save_mean_file。
#    def mean(self,to_mean_file,save_mean_file):
#        data=pd.read_csv(to_mean_file,header=None,index_col=None)
#        data_mean=data.mean(1)  
#        data_mean.to_csv(save_mean_file,header=False,index=False)
#        print("Generate "+str(save_mean_file)+".")
#        return 0
  
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
    
    #对输入back数据按行求均值，生成一列。
    def back_mean(self):
        #back_mean_file=self.back_mean_data
        data=pd.read_csv(self.back_cut_data,header=None,index_col=None)
        data_mean=data.mean(1)  
        data_mean.to_csv(self.back_mean_data,header=False,index=False)
        print("Generate back_mean_data.csv.")
        return 0
    
#        #对输入back数据按行求均值，生成一列。
#    def back_mean(self):
#        self.mean_by_category(self.back_cut_data,1,self.back_mean_data)
#        print("Generate back_mean_data.csv.")
#        return 0
        
    #批量产生所有类型(裁剪ed,去背景ed,正则化ed)的sample的均值列，供画图使用(不参与余下计算)。
    def all_kinds_sample_mean(self,category):
        self.mean_by_category(self.sample_cut_data,category,self.sample_mean_data)
        print("Generate sample_mean_data.csv.")
        self.mean_by_category(self.sample_debacked_data,category,self.sample_debacked_mean_data)
        print("Generate sample_debacked_mean_data.csv.")
        self.mean_by_category(self.sample_scaled_data,category,self.sample_scaled_mean_data)
        print("Generate sample_scaled_mean_data.csv.\n")
        return 0   
    
#------------------------------5.4线性插值算法----------------------------------
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
                
#--------------------5.5针对去背景方式的不同，创建插值背景算法--------------------
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
            interpolation_num=sample_data.shape[1]/SampleInformation().class_num
        #若采用传统的手动放置样本，则直接整批样本去插值背景
        else:
            interpolation_num=sample_data.shape[1]
        #开始插值，起始，结尾，插值点数，保存名
        self.linear_interpolation(self.mean_start_back,self.mean_end_back,interpolation_num,self.interpolation_back)
        return 0
    
#--------------------------------5.4去背景-------------------------------------
    #样本普去背景。  
    def deback(self,interpolation_back=False):
        #读入待处理.csv文件
        data=pd.read_csv(self.sample_cut_data,header=None,index_col=None)
        #若开启去插值背景的去背景方式，则读入插值背景；否则读入均值背景
        per_sample_num=data.shape[1]/SampleInformation().class_num
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

    #通用样本普去背景，任意指定样本谱和背景,采用最常用的去均值背景。  
    def universal_deback_mean(self,sample_data,back_data,debacked_data):
        #读入待处理.csv文件
        data=pd.read_csv(sample_data,header=None,index_col=None)
        back=pd.read_csv(back_data,header=None,index_col=None)
        #创建deback的Dataflame
        deback=pd.read_csv(sample_data,header=None,index_col=None)
        #行row=j，列column=i
        i=j=0
        #back_col=0
        #获取行数data.shape[0]，获取列数data.shape[1]
        while i<data.shape[1]:
            while j<data.shape[0]:
                #启动去均值背景的去背景方式
                #判断back或data中有0值，则log运算会为±无穷，出错，因此直接置零，跳过log
                if back.iloc[j,0]==0 or data.iloc[j,i]==0:
                    deback.iloc[j,i]=0
                else:
#                    deback.iloc[j,i]=math.log(back.iloc[j,back_col]/data.iloc[j,i])
                    deback.iloc[j,i]=math.log(back.iloc[j,0]/data.iloc[j,i])
                j+=1
            j=0
            i+=1
        #print(deback)
        deback.to_csv(debacked_data,header=False,index=False)
        print("Generate debacked_data.csv based on mean_back.csv")
        return 0

#-----------------------------5.5正则化----------------------------------------     
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
    
    #针对实验样本名称，调用上面通用标准化方法封装好，可选择是标准化（default）还是归一化
    def scale(self,scaled_to_01=False):
        self.data_normalization(self.sample_debacked_data,self.sample_scaled_data,scale_to_01=scaled_to_01)
        print("Generate sample_scaled_data.csv")
        return 0
    
#------------------5.6重命名输出back.csv和data.csv备用--------------------------    
    #将经过以上所有处理步骤后的最终文件，重新命名为back.csv和data.csv，便于后续调用。
    def finally_data(self):
        back=pd.read_csv(self.back_mean_data,header=None,index_col=None)
        back=pd.DataFrame(back)
        back.to_csv(self.finally_back_data,header=None,index=None)
        data=pd.read_csv(self.sample_scaled_data,header=None,index_col=None)
        data=pd.DataFrame(data)
        data.to_csv(self.finally_sample_data,header=None,index=None)
        print("Generate back.csv!")
        print("Generate data.csv!")
        return 0
   
    
class CreateLabel():
    """6.根据样本数和分类数自动创建标签文件"""
    #样本数可自动获取，分类数需手动输入
    def __init__(self,category_num):
        self.category_num=category_num
        self.label_name=DataNameSet().label_data
        
    def get_sample_num(self,):
        sample=pd.read_csv(DataNameSet().sample_data,header=None,index_col=None)
        sample_num=sample.shape[1]
        return sample_num
    
    #包含独热编码和标签编码两种常用的编码方式（一般来说独热编码比较常用）
    #但是用scikit-learn的confusion_matrix画多元混淆矩阵时，目前只能用标签编码，才不会报错
    #默认采用独热编码方式进行编码，若输入'LabelEncoder'，则采用标签编码    
    def encoder(self,encoder_mode="OneHotEncoder"):
        #获取样本数信息
        sample_num=self.get_sample_num()#需加括号，否则认为是调用方法，而非接受其返回值。
        #创建分类数大小的单位矩阵
        Inum=np.identity(self.category_num)
        #每类样本的数量num_per_sam
        num_per_sam=int(sample_num/self.category_num)
        label=[]
        row=col=0
        while row<self.category_num:
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
        label.to_csv(self.label_name,header=False,index=False)
        print("Generate label.csv!\n")
        return 0 
    

class DReduction():
    """7.创建用于降维的各种方法,如PCA等，后续可拓展使用LDA等方法"""
    def __init__(self,components):
        self.components=components
        self.data_set_name=DataNameSet().finally_sample_data
        self.Dred_data=DataNameSet().Dre_data
        
    #PCA降维算法。
    def Pca(self):
        data_set=pd.read_csv(self.data_set_name,header=None,index_col=None)
        data_set=data_set.T   
        pca=PCA(n_components=self.components)
        pca.fit(data_set)
        data_set=pca.transform(data_set)
        print("Generate Pca_data.csv." )
        print("The interpretability of each component:")
        print(pca.explained_variance_ratio_)
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(self.Dred_data,header=False,index=False)
        return 0
    
    
    #PCA通用降维算法。
    def universal_pca(self,csv_name,pcaed_name):
        data_set=pd.read_csv(csv_name,header=None,index_col=None)
        data_set=data_set.T   
        pca=PCA(n_components=self.components)
        pca.fit(data_set)
        data_set=pca.transform(data_set)
        print("Generate Pca_data.csv." )
        print("The interpretability of each component:")
        print(pca.explained_variance_ratio_)
        data_set=pd.DataFrame(data_set)
        data_set.to_csv(pcaed_name,header=False,index=False)
        return 0
    
    
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
        
from sklearn.model_selection import cross_val_score      
class MLFrame():
    """8.定义整套抽象机器学习算法框架，包含{splitting，training、evaluation}三大部分"""
    """对于具体机器学习算法，只需继承该类，并重写training部分即可。"""
    def __init__(self,):
        #splitting输入数据集属性
        self.data=DataNameSet().Dre_data
        self.label=DataNameSet().label_data
        #training时的中间属性
        self.x_train=self.x_test=self.y_train=self.y_test=[]        
        #通过training输出的属性==evaluation时输入的属性（四大属性）
        #分为三大类：输出精度，基于混淆矩阵的评估、画ROC和AUC
        self.train_accuracy=0#用于输出训练精度
        self.test_accuracy=0#用于输出测试精度
        self.y_predict=[]#用于画混淆矩阵
        self.y_predict_proba=[]#用于画ROC和AUC
        #输入CT重建后的光谱值，该值已经去背景、归一化、降维后，与Dre_data相似，用于预测
        self.CT_x_data=DataNameSet().CT_x_data
        
        
#========================8.1splitting,数据集划分================================
    #（这里可选是否将标签转为独热编码，因此前面创建标签时一般都创建成标签编码格式）   
    def split_data_set(self,binarize_label=False):
        #读取pca降维后的数据集、对应标签
        data=pd.read_csv(self.data,header=None,index_col=None)
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
            label=label_binarize(label,classes=list(range(0,SampleInformation().class_num)))   
        #数据集分割成训练集和测试集(后续考虑采用k折交叉验证法分割数据集)。
        #测试集占20%，shuffle选择是否打乱顺序（不打乱则效果很差）
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(
                data,label,test_size=0.2,shuffle=True)
        return 0

#=====================8.2training，训练拟合部分，留空===========================    
    def classifiers(self):
        pass
        return 0

#======================8.3evaluation，评估部分，分四类==========================
#------------------------8.3.1输出训练精度、测试精度-----------------------------
    def output_accuracy(self,):
        print("\n1.训练集和测试集准确率：")
        print("Training accuracy: %f" % self.train_accuracy)
        print("Test accuracy: %f" % self.test_accuracy)
        return 0
    
#---------------------8.3.2基于混淆矩阵评估，标签编码----------------------------    
    #都是对y_test和y_predict做对比,核心为混淆矩阵。   
    def confusion_matrix_eval(self,):
        #1.多元混淆矩阵
        cm=confusion_matrix(self.y_test,self.y_predict) 
        print("\n3.多元混淆矩阵：")
        print(cm)
        #2.自动评估报告
        #target_names=['class0','class1','class2','class3','class4','class5']
        #统一到DatanameSet()中管理分类样本名称。
        cr=classification_report(self.y_test,self.y_predict,
                                 target_names=SampleInformation().class_names)
        print("\n4.自动评估报告：")
        print(cr)
        #3.cohen-kappa系数
        ck=cohen_kappa_score(self.y_test,self.y_predict)
        print("\n5.cohen-kappa系数：")
        print(ck)
        #4.马修斯相关性系数(MCC)
        mcc=matthews_corrcoef(self.y_test,self.y_predict)
        print("\n6.马修斯相关性系数(MCC)：")
        print(mcc)
        return 0
     
    #绘制彩色的混淆矩阵，正则化可选，default为非正则化
    def draw_confusion_matrix(self, normalize=False,cmap=plt.cm.Blues):
        cm=confusion_matrix(self.y_test,self.y_predict) 
        classes=SampleInformation().class_names
        if normalize:
            #正则化
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
            #print("\n7.Normalized confusion matrix")
        else:
            title='Confusion matrix without normalization'
            print('\n7.Confusion matrix with and without normalization')   
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
        plt.savefig('graph\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.figure()
    
#-------------------------8.3.3画ROC和AUC曲线，独热编码-------------------------
    def draw_ROC_AUC(self,):
        print('\n8.ROC and AUC curves:')
        #由于sklearn中混淆矩阵和ROC只能分别用标签编码和独热编码，因此目前只能在这里重新拟合计算一遍
        self.split_data_set(binarize_label=True)
        self.classifiers()
        #self.generating_evaluation_parameters()
        y_score=self.y_predict_proba
        #print(y_score)
        y_test=self.y_test
        n_classes=SampleInformation().class_num
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
                 label='micro-AVG({0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-AVG({0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='Class{0}({1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of ROC to multi-class')
        plt.legend(loc="lower right")
        #保存图片
        title='ROC_curve'
        plt.savefig('graph\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.show()    

#-------------------------8.3.4画PR曲线，独热编码-------------------------   
    #绘制PR曲线
    def draw_PR_curve(self,):
        print("\n9.P-R曲线：")
        #计算平均查准率(仅用于2分类情况)
#        y_test=self.y_test
#        y_score=self.y_predict
#        average_precision = average_precision_score(y_test, y_score)
#        print('Average precision-recall score: {0:0.2f}'.format(
#              average_precision))
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
        ###绘制2分类PR曲线
#        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#                  average_precision))
        #绘制多分类PR曲线：
        #重新传递参数（由于需要独热编码，所以应重新计算，跟前面ROC绘制同样道理）
        self.split_data_set(binarize_label=True)
        self.classifiers()
        y_score=self.y_predict_proba
        Y_test=self.y_test
        n_classes=SampleInformation().class_num
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
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))
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
        plt.savefig('graph\\'+title+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+title+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        
#----------------------------8.3.5交叉验证评估----------------------------------
    #独立定义交叉验证评估方式，需借助classifier参数来评估： 
    #默认折数为5，可自定义，当折数等于样本数时，变成留一法
    def cross_validation(self,k_fold=5):
        #读取降维后的数据集、对应标签
        data=pd.read_csv(self.data,header=None,index_col=None)
        label=pd.read_csv(self.label,header=None,index_col=None)
        label=label.T  
        #传入的label只需1维格式
        label=np.array(label).ravel()
        #classifiers()中已返回classifier参数
        clf=self.classifiers()
        scores=cross_val_score(clf,data,label,cv=k_fold)
        print("\n10.K折交叉验证各折准确率:")
        print(scores)
        print("\n11.K折交叉验证平均准确率及置信区间:")
        print("Average accuracy of CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
#    def generating_evaluation_parameters(self,):
#        clf=self.classifiers()
#        #无论用什么机器学习算法训练，都必须输出以下四大参数供评估使用
#        self.y_predict=clf.predict(self.x_test)
#        self.y_predict_proba=clf.predict_proba(self.x_test)  
#        self.train_accuracy=clf.score(self.x_train,self.y_train)
#        self.test_accuracy=clf.score(self.x_test,self.y_test)
#        return 0
        
#---------------------------8.3.6对CT重建进行预测-------------------------------
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
         
#======8.4综合以上{splitting，training、evaluation}三部分函数，便于直接调用=======
    #也可在具体机器学习算法中重写该部分函数，进行特定的评估
    #该部分函数应通过具体机器学习算法来继承调用，而不能直接调用(因父类classifier为空)
    def training(self,):
        self.split_data_set()
        #将分类器训练过程和评估参数生成过程分离，使后续重构更简单
        self.classifiers() 
        return 0
        
    def evaluation(self,K_Fold=5):
        self.output_accuracy()
        self.CT_predict()
        self.confusion_matrix_eval()
        self.draw_confusion_matrix()
        self.draw_confusion_matrix(normalize=True)
        self.draw_ROC_AUC() 
        self.draw_PR_curve()
        self.cross_validation(k_fold=K_Fold)
        return 0
    

class NeuralNetwork(MLFrame):
    """9A.神经网络算法，继承MLFrame父类，重写ML框架中的classifier()函数即可使用"""
    """不同机器学习算法其实只是分类器classifier不同，因此重写该方法即可"""
    def __init__(self):
        #继承父类的所有属性和方法
        super(NeuralNetwork,self).__init__()
    #重写父类中的classifiers方法    
    def classifiers(self):
        classifier=MLPClassifier(hidden_layer_sizes=(100,100),
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
    """9B.支持向量机算法(Support Vector Machine，SVM)"""
    def __init__(self):
        super(SVM,self).__init__()
   
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
    """9C.K近邻算法"""
    def __init__(self,):
        super(KNeighbors,self).__init__()
    
    def classifiers(self):
        classifier=KNeighborsClassifier(n_neighbors=5,
                                        weights='uniform',
                                        algorithm='auto',
                                        leaf_size=30,p=2,
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

    
class Drawing(DataNameSet):
    """10.定义绘图相关属性和方法，包含{背景图X1、光谱图X3、PCA图X3}三部分，共7张"""
    #统一定义图形相关参数，统一命名图片存储名称
    def __init__(self,):
        super().__init__()
        self.colors=['b','c','g','k','m','r','y','w','b','c']
        self.markers=['o','v',',','*','x','s']
        self.line_type=['k','k-','k--','k:','k','k-',]
        self.sample_names=SampleInformation().class_names
        self.back_name=('Background spectrum',)#逗号不能少
        #所有图片的命名
        self.back_pic="back"
        self.sample_pic="each_sample"
        self.sample_mean_pic="sample_mean"
        self.sample_debacked_pic="sample_debacked_mean"
        self.sample_scaled_pic="sample_scaled_mean"
        self.sample_pca2D_pic="pca2D_scatter"
        self.sample_pca3D_pic="pca3D_scatter"
        self.sample_pca_pic="pca_scatter"
        
#=================10.1定义通用绘图函数，用于绘制背景图和光谱图====================     
    def draw_data(self,csv_name,title,xlabel="Channel",ylabel="Number of photons",save_name="test",cnum=0):
        #读入csv数据
        data=pd.read_csv(csv_name,header=None,index_col=None)
        i=0
        x = np.arange(data.shape[0]) 
        #print(data.shape[1])
        
        if data.shape[1]==1:
            #说明是back_data
            y=data.iloc[:,0]
            #plt.plot(x,y,self.line_type,c=self.colors[i],marker=self.markers[i],linewidth=1,) 
            #plt.plot(x,y,self.line_type,c=self.colors[i],linewidth=1,) 
            plt.plot(x,y,linewidth=1,) 
            plt.legend(self.back_name)
        else:
            #说明是sample_data
            while i<data.shape[1]:#获取列数
                y=data.iloc[:,i]
                if cnum==0:
                    #说明是平均值后的sample_data，则直接画
                    #plt.plot(x,y,self.line_type,c=self.colors[i],marker=self.markers[i],linewidth=1,) 
                    #plt.plot(x,y,self.line_type,c=self.colors[i],linewidth=1,) 
                    #plt.plot(x,y,self.line_type[i],linewidth=1,) 
                    plt.plot(x,y,linewidth=1,) 
                    i+=1
                else:
                    #否则是画所有的sample_data，同一类的画同一种颜色
                    per_num=data.shape[1]/cnum
                    j=int(i/per_num)
                    #print(j)
                    plt.plot(x,y,c=self.colors[j],linewidth=1,)
                    i+=1
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
        plt.savefig('graph\\'+save_name+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+save_name+'.png',bbox_inches='tight',dpi=128)
        plt.show()
        return 0

#======================10.2定义调试时的绘图工具，可画指定列=======================    
    #能画出指定文件的指定列数据，可同时在一图里画多列，cols指定第几列，可接受无数多个列
    #该函数用于MLDemo中，用于分析数据时，灵活画图观察使用。
    def draw_specific_col(self,csv_name,save_name="test",*cols):
        title="draw particular columns for analyse "
        xlabel="Channel"
        ylabel="Number of photons"
        #读入csv数据
        data=pd.read_csv(csv_name,header=None,index_col=None)
        x = np.arange(data.shape[0]) 
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
   
#=======================10.3绘制1种背景图和3种光谱图============================
    def draw_back_and_samples(self):
        #画出平均背景数据
        self.draw_data(self.back_mean_data,
                       "Background spectrum","Channel","Number of photons",
                       self.back_pic)
        #画所有样本数据，需提供样本类数信息，便于按类分颜色画出
        self.draw_data(self.sample_cut_data,
                       "All samples XAS","Channel","Number of photons",
                       self.sample_pic,SampleInformation().class_num)
        
        self.draw_data(self.sample_debacked_data,
                       "All samples XAS","Channel","Number of photons",
                       "sample_debacked_data")
        
                       
        #画出平均值类型的样本数据
        self.draw_data(self.sample_mean_data,
                       "Samples mean XAS","Channel","Number of photons",
                       self.sample_mean_pic)
        self.draw_data(self.sample_debacked_mean_data,
                       "Samples debacked mean XAS","Channel","Number of photons",
                       self.sample_debacked_pic)
        self.draw_data(self.sample_scaled_mean_data,
                       "Samples scaled mean XAS","Channel","Number of photons",
                       self.sample_scaled_pic)
        return 0

#============================10.4绘制3种PCA图==================================
    #取前2主成分画二维图    
    def draw_pca_2D(self,category_num):
        data=pd.read_csv(self.Dre_data,header=None,index_col=None)
        #each_num：每类样本个数
        each_num=int(data.shape[0]/category_num)
        x, y = data[0], data[1]
        i=0
        while i<category_num:
            s_row=i*each_num
            e_row=((i+1)*each_num)-1
            #plt.scatter(x[s_row:e_row], y[s_row:e_row], c=self.colors[i],marker=self.markers[i])
            plt.scatter(x[s_row:e_row], y[s_row:e_row],)
            i+=1
        #plt.title('PCA',fontsize=16)
#        ax.set_zlabel('PCA1',fontsize=6) 
#        ax.set_ylabel('PCA2',fontsize=6)
#        ax.set_xlabel('PCA3',fontsize=6)
        #设置刻度标记大小labelsize
        plt.tick_params(axis='both',which='major',labelsize=10)
        #自动保存图片
        plt.savefig('graph\\'+self.sample_pca2D_pic+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+self.sample_pca2D_pic+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        return 0   
        
    #取前3主成分画三维图    
    def draw_pca_3D(self,category_num):
        data=pd.read_csv(self.Dre_data,header=None,index_col=None)
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
        ax.set_zlabel('PCA1',fontsize=6) 
        ax.set_ylabel('PCA2',fontsize=6)
        ax.set_xlabel('PCA3',fontsize=6)
        #设置刻度标记大小labelsize
        plt.tick_params(axis='both',which='major',labelsize=10)
        #自动保存图片
        save_name=self.sample_pca_pic
        plt.savefig('graph\\'+save_name+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+save_name+'.png',bbox_inches='tight',dpi=256)
        plt.show()
        return 0    
    
    #取前4主成分画散点图
    def draw_pca_scatter(self):
        data=pd.read_csv(self.Dre_data,header=None,index_col=None)
        title=["PCA1","PCA2","PCA3","PCA4"]
        i=0
        x = np.arange(data.shape[0]) 
        while i<4:
            plt.subplot(221+i)
            plt.scatter(x,data.iloc[:,i],s=1)
            plt.title(title[i])
            i+=1
            plt.grid(True)
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, 
                            hspace=0.35,wspace=0.25)
        plt.legend(self.sample_names)
        save_name=self.sample_pca3D_pic
        plt.savefig('graph\\'+save_name+'.pdf',bbox_inches='tight',)
        plt.savefig('graph\\'+save_name+'.png',bbox_inches='tight',dpi=256)
        plt.show()        
        return 0  
    
    


            
    