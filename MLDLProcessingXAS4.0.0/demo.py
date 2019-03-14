#!/bin/env python 3.6
# -*- encoding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2018-06-22
# Version:     2.1.0
# E-mail:      674649741@qq.com
# Purpose:     分析数据时可以临时画出指定csv的指定列数据
#==============================================================================
from mlAPI import Drawing
dr=Drawing()

from mlAPI import CSVProcessing
cp=CSVProcessing()

from mlFunctions import MLCsvProcessing
mcp=MLCsvProcessing()

from mlAPI import CSVGenerating
cg=CSVGenerating()


#cg.txt_to_csv('\\sl_rebuilt_XAS','file_output/all_data.csv')

#=============================================================================

#cg.txt_to_csv('\\4铝板实验数据\\30kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\30kv5ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\30kv10ua','file_output/all_data.csv')

#cg.txt_to_csv('\\4铝板实验数据\\40kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\40kv5ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\40kv10ua','file_output/all_data.csv')

#cg.txt_to_csv('\\4铝板实验数据\\50kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\50kv5ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\50kv10ua','file_output/all_data.csv')

#cg.txt_to_csv('\\4铝板实验数据\\60kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\60kv5ua','file_output/all_data.csv')
#cg.txt_to_csv('\\4铝板实验数据\\60kv10ua','file_output/all_data.csv')

#cg.txt_to_csv('\\4铝板实验数据\\前后60kv2ua空气组一致性比对','file_output/all_data.csv')
#dr.draw_specific_col('file_output/all_data.csv','back',1,2,3,4,5,6)
#=============================================================================


#=============================================================================
#cg.txt_to_csv('\\5仙草冻实验数据\\30kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\30kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\30kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\5仙草冻实验数据\\40kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\40kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\40kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\5仙草冻实验数据\\50kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\50kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\50kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\5仙草冻实验数据\\60kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\60kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\60kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\5仙草冻实验数据\\测试不同空瓶一致性','file_output/all_data.csv')
#cg.txt_to_csv('\\5仙草冻实验数据\\测试原液稀释3.4倍不摇匀是否产生影响','file_output/all_data.csv')

#dr.draw_specific_col('file_output/all_data.csv','back',1,2,3,4,5,6)
#=============================================================================


#=小鼠肝脏实验==================================================================
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\30kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\30kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\30kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\40kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\40kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\40kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\50kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\50kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\50kv6ua','file_output/all_data.csv')

#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\60kv2ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\60kv4ua','file_output/all_data.csv')
#cg.txt_to_csv('\\6小鼠肝硬化实验181210\\60kv6ua','file_output/all_data.csv')

#dr.draw_specific_col('file_output/all_data.csv','back','30kv2ua',1,2,3,)
#=============================================================================



#=====计算相对差
#import pandas as pd
#import numpy as np
#data=pd.read_csv('file_output/all_data.csv',header=None,index_col=None)
#data_1=np.asarray(data.iloc[:,2])
#index_1=np.argmax(data_1,axis=0)
##print(data,data.shape,index_1)
#val=(data.iloc[index_1,1]-data.iloc[index_1,2])/(data.iloc[index_1,1]+data.iloc[index_1,2])
#print('value:',val)




#=====去背景===================================================================

#dr.draw_specific_col('file_temp/sample_3_debacked.csv','data','The XAS of ten plastics',1,2,3,4,5,6,7,8,9,10)
#dr.draw_specific_col('file_temp/sample_4_scaled.csv','data','The XAS of ten plastics after scaled',1,2,)
#dr.draw_specific_col('file_temp/sample_4_scaled.csv','data','The XAS of UV after scaled',10,)

#dr.draw_specific_col('file_temp/sample_3_debacked.csv','data','60kv10ua',1,2,3,4,5)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
file='file_temp/sample_3_debacked.csv'
data=pd.read_csv(file,header=None,index_col=None)
data=np.asarray(data)
x=list(range(data.shape[0]))
for d in range(data.shape[1]):
#    y=data[:,d]
#    print(y)
    plt.plot(x,data[:,d])
plt.show()
    


#==============================================================================


#=====画10种塑料===============================================================
#import scipy.io as sio
#import os
#import numpy as np
#import pandas as pd
##name='UV'
###载入7种mat格式的塑料透射光谱
#def load_mat():
#    file="0塑料CT相关数据\\塑料实验数据7种mat\\PVDF.mat"
#    data=sio.loadmat(file)
#    data=data['PVDF']
#    print(data)
#    data=pd.DataFrame(data)
#    data.to_csv('data.csv',header=None,index=None)
#    print(file)
#    all_data=[]
#    for file in files:
#        data=sio.loadmat('0塑料CT相关数据\\塑料实验数据7种mat\\'+file)
#        print(file)
#        name=file[:-4]
#        print(name)
#        data=pd.DataFrame(data[name])
##        data=data.mean(1)
##        data=data.iloc[:,0]
##        print(data)
#        all_data.append(data)
#    print(all_data,len(all_data))
#    all_data=pd.DataFrame(all_data)
#    all_data=all_data.T
#    all_data.to_csv('all_data_7.csv',header=None,index=None)
    
#def load_mat():
#    files=os.listdir("0塑料CT相关数据\\塑料实验数据7种mat")
#    print(files)
#    all_data=[]
#    for file in files:
#        data=sio.loadmat('0塑料CT相关数据\\塑料实验数据7种mat\\'+file)
#        print(file)
#        name=file[:-4]
#        print(name)
#        data=pd.DataFrame(data[name])
##        data=data.mean(1)
##        data=data.iloc[:,0]
##        print(data)
#        all_data.append(data)
#    print(all_data,len(all_data))
#    all_data=pd.DataFrame(all_data)
#    all_data=all_data.T
#    all_data.to_csv('all_data_7.csv',header=None,index=None)
    
#load_mat()

#def load_csv():
#    names=['back','SG','TAR','UV']
#    all_data=[]
#    for name in names:
#        data=pd.read_csv(name+'.csv',header=None,index_col=None)
#        data=data.mean(1)
#        print(data)
#        all_data.append(data)
#    print(all_data,len(all_data))
#    all_data=pd.DataFrame(all_data)
#    all_data=all_data.T
#    all_data.to_csv('all_data_back_and_3.csv',header=None,index=None)
#        
#load_csv()
#
##将十种塑料放在一起进行归一化
#from sklearn.preprocessing import MinMaxScaler
#def scale_plastics():
#    data=pd.read_csv('mean_back_and_plastics.csv',header=None,index_col=None)
#    pl_data=data.iloc[:,1:]
##    back_data=data.iloc[:,0]
#    print(pl_data,pl_data.shape)
#    mms=MinMaxScaler()
#    pl_data=mms.fit_transform(pl_data)
#    pl_data=pd.DataFrame(pl_data)
#    pl_data.to_csv('mean_scaled_back_and_plastics.csv',header=None,index=None)
#    
#scale_plastics()
    
    
        

#        
#import pandas as pd
#import math
##data=pd.DataFrame(data[name])
#
##data=pd.read_csv('塑料实验原始数据\\SG.csv',header=None,index_col=None)
#data=pd.read_csv('ten_plastics.csv',header=None,index_col=None)
#print(data)
#
#def cut(data,s_row,e_row):
#    data=data.iloc[s_row:e_row,:]
#    print(data.shape)
#    return data
#
#def mean_back(back):
#    back_mean=back.mean(1)
#    return back_mean
#
#def deback(data,back):
#    i=j=0
#    while i<data.shape[1]:#i为列数
#        while j<data.shape[0]:#j为行数
#            data.iloc[j,i] = math.log(back.iloc[j,0] / data.iloc[j,i])
#            j+=1
#        i+=1
#        j=0
#    return data
#    
#def processing(data):
#    data=cut(data,s_row=7,e_row=258)
#    back=pd.read_csv('back/back_2.csv',header=None,index_col=None)
#    back=cut(back,s_row=7,e_row=258)
#    back=mean_back(back)
#    back.to_csv('back.csv',header=None,index=None)
#    back=pd.read_csv('back.csv',header=None,index_col=None)
#    data=deback(data,back)
#    print(data,back)
#    data.to_csv('file_output/all_data.csv',header=None,index=None)
#    dr.draw_specific_col('file_output/all_data.csv','debacked','ten plastics',1,2,3,4,5,6,7,8,9,10)
#
#if __name__=='__main__':
##    data=pd.read_csv('ten_plastics.csv',header=None,index_col=None)
#    processing(data)
#=============================================================================


#=============================================================================
#dr.draw_specific_col("file_temp\\sample_3_debacked.csv","test",'50KV2UA',1,2,3,4,5,)
#dr.draw_specific_col("file_temp\\sample_4_scaled.csv","test",1,2,3,4,5,6,7,8,9,10)
#dr.draw_specific_col("file_temp\\sample_4_scaled.csv","test",1,2,3,4,5,6,7,8,9,10)
#============================================================================


#draw_specific_col:{输入文件；保存图片名称；绘制哪几列(数量不限)}
#dr.draw_specific_col("file_temp\\sample_3_debacked.csv","test",1,2,3,4,5)

#mcp.get_back_smart([1,10],[11,20])

#import pandas as pd
#import os
#
#a={}
#a=pd.DataFrame(a)
#a.to_csv("file_temp\\testing.csv",header=None,index=None)
#
#if os.path.exists("file_temp\\testing.csv"):
#    print("YES")
#    os.remove("file_temp\\testing.csv")

#cp.split_combine_data("file_output\\load_data.csv","file_temp\\temp.csv",[1,10],[11,20])