# -*- coding: utf-8 -*-
#import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import ploting
import data_preprocessing as dpp
import data_information as di
import prediction
fns=di.FileNameSetting()
disi=di.SampleInformation()
import training
import os
import re
from mlFunctions import MLCsvProcessing
mlcp=MLCsvProcessing()
#import sys
#command对应的function不能加（），否则没按下就开始运行了
###############################################################################
#定义各类事件
#信息导入按钮    
def button_event_1():
#    root.withdraw()
    #传入样本位置
    dpp.txt_dir=entry_2_1.get()
    #传入样本数
    di.global_class_num=int(entry_2_2.get())
    #print(di.global_class_num)
    #传入样本名
    sample_names=entry_2_3.get()
    di.global_class_names=sample_names.split(',')
    #判断样本数和样本名是否相等
    if di.global_class_num != len(di.global_class_names):
        auto_name=disi.auto_create_name(entry_2_2.get())
        entry_txt_2_3.set(auto_name)
        roll_text_1.insert('end',"样本类数和样本名称数不匹配，已自动补齐!\n")
    #print(di.global_class_names)
    #传入back列
    back_data=entry_2_4.get()
#    dpp.back_data_s=int(back_data.split(',')[0])
#    dpp.back_data_e=int(back_data.split(',')[1])
#    #测试
    back_data=mlcp.convert_str_to_list(back_data)
    dpp.back_data=back_data
    #print(back_data)
    
    #出入sample列
    sample_data=entry_2_5.get()
#    dpp.sample_data_s=int(sample_data.split(',')[0])
#    dpp.sample_data_e=int(sample_data.split(',')[1])
    sample_data=mlcp.convert_str_to_list(sample_data)
    dpp.sample_data=sample_data
    
    
    #打印状态信息
    if di.global_class_num == len(di.global_class_names):
        roll_text_1.insert('end',"\n导入信息成功!\n")
    return 0

#数据处理按钮
def button_event_2():
    #防止未导入信息而报错
    if di.global_class_names == 'null':
        roll_text_1.insert('insert',"\n请先导入信息，再进行模型评估！")
    else:
        roll_text_1.insert('insert',"\n数据处理中，请稍等...")
        roll_text_1.update()
        #传入通道截取行
        channel=entry_2_6.get()
        dpp.channel_s=int(channel.split(',')[0])
        dpp.channel_e=int(channel.split(',')[1])
        #判断复选框状态，确定dpp.main()中是否全部运行
        #裁剪
        if cb_v1.get() == 0:
            dpp.botton_state[0]=0
        else:
            dpp.botton_state[0]=1   
        #均值
        if cb_v2.get() == 0:
            dpp.botton_state[1]=0
        else:
            dpp.botton_state[1]=1
        #去背景
        if cb_v3.get() == 0:
            dpp.botton_state[2]=0
        else:
            dpp.botton_state[2]=1
        #数值缩放
        if cb_v4.get() == 0:
            dpp.botton_state[3]=0
        else:
            dpp.botton_state[3]=1
        #降维
        if cb_v5.get() == 0:
            dpp.botton_state[4]=0
        else:
            dpp.botton_state[4]=1
        #生成标签
        if cb_v6.get() == 0:
            dpp.botton_state[5]=0
        else:
            dpp.botton_state[5]=1  
        #处理降维保留维数信息
        dpp.dred_num=int(entry_2_11.get())
        #print(dpp.dred_num)
        #处理下拉框状态
        #根据下拉框3的状态判断生成什么样的背景，以及如何去背景    
        dpp.com_1_back_state=combobox_1.get()
        dpp.com_2_scale_state=combobox_2.get()
        dpp.com_3_dre_state=combobox_3.get()
        #print(dpp.com_3_dre_state)
        #dpp.com_4_model_state=combobox_4.get()
    #    combobox_2.current(1)
        #开始处理数据
        dpp.main()
        #反馈处理后结果
        roll_text_1.insert('insert',"\n  ①.生成back、sample、label文件；")
        roll_text_1.insert('insert',"\n  ②.前"+str(dpp.dred_num)+"个主成分贡献率:\n")
        roll_text_1.insert('insert',str(dpp.pca_ret[0]))
        roll_text_1.insert('insert',"\n  ③.累计贡献率:")
        roll_text_1.insert('insert',str(dpp.pca_ret[1]))
        roll_text_1.update()
        roll_text_1.insert('end',"\n完成数据处理！\n")
    return 0

#按钮事件3，开始训练
def button_event_3():
    #选取模型
    dpp.com_4_model_state=combobox_4.get()
    #传入训练集和测试集比例
    di.test_set_ratio=float(entry_2_10.get())
    roll_text_1.insert('end',"\n开始训练...")
    #开始训练
    training.train()
    #窗口反馈训练结果
    roll_text_1.insert('end',"\n  ①.Training accuracy="+str(training.output_accuracy[0]))
    roll_text_1.insert('end',"\n  ②.Test accuracy="+str(training.output_accuracy[1])+"\n")
    roll_text_1.insert('end',"\n  ③.模型权重值为："+str(training.model.coefs_))
    roll_text_1.insert('end',"\n  ④.模型偏置值为："+str(training.model.intercepts_))
    roll_text_1.insert('end',"\n训练完成！\n")
    return 0


#按钮事件4，模型评估
def button_event_4():
    #防止未导入信息而报错
    if di.global_class_names == 'null':
        roll_text_1.insert('insert',"\n请先导入信息，再进行模型评估！")
    else:
        #根据复选框选择评估方式,传给data_information,再传给mlAPI
        if cb_v7.get() == 0:
            #CM(混淆矩阵)
            di.evaluation_botton_state[0]=0
        else:
            di.evaluation_botton_state[0]=1
        if cb_v8.get() == 0:
            #ROC
            di.evaluation_botton_state[1]=0
        else:
            di.evaluation_botton_state[1]=1
        if cb_v9.get() == 0:
            #PRC
            di.evaluation_botton_state[2]=0
        else:
            di.evaluation_botton_state[2]=1
        if cb_v10.get() == 0:
            #CV
            di.evaluation_botton_state[3]=0
        else:
            di.evaluation_botton_state[3]=1
        #若都不选定，给出提示
        if di.evaluation_botton_state == [0,0,0,0]:
            roll_text_1.insert('end',"\n未选中任何评估方式！")
        #传入K折交叉验证折数
        training.evaluation_cv_fold=int(entry_2_12.get())
        #提示模型正在评估
        roll_text_1.insert('insert',"\n模型评估中，请稍等...")
        roll_text_1.update()
        #开始评估
        evaluation_result=training.evaluation()
        roll_text_1.insert('end',"\n评估结果如下：")
        #反馈评估结果
        #这边涉及到嵌套列表的输出方式
        if cb_v7.get() == 1:
        #CM相关
            roll_text_1.insert('end',"\n  ①.混淆矩阵：")
            roll_text_1.insert('end',"\n"+str(evaluation_result[0][0]))
            roll_text_1.insert('end',"\n  ②.评估报告：")
            roll_text_1.insert('end',"\n"+str(evaluation_result[0][1]))
            roll_text_1.insert('end',"  ③.Cohen-Kappa系数：")
            roll_text_1.insert('end',str(evaluation_result[0][2]))
            roll_text_1.insert('end',"\n  ④.MCC系数：")
            roll_text_1.insert('end',str(evaluation_result[0][3]))
        if cb_v10.get() == 1:
        #CV相关
            roll_text_1.insert('end',"\n  ⑤"+str(training.evaluation_cv_fold)+"折交叉验证各折准确率：\n")
            roll_text_1.insert('end',str(evaluation_result[1][0]))
            roll_text_1.insert('end',"\n  ⑥.平均准确率：")
            roll_text_1.insert('end',str(evaluation_result[1][1]))
            roll_text_1.insert('end',"\n  ⑦.对应置信区间(+/-)：")
            roll_text_1.insert('end',str(evaluation_result[1][2]))
        if cb_v8.get() == 1 or cb_v9.get() ==1:  
        #ROC,PRC相关
            roll_text_1.insert('end',"\nROC、PRC结果请打开文件夹查看;")
        roll_text_1.insert('end',"\n评估完成！\n")
    return 0

#按钮事件5，数据分析，即ploting
def button_event_5():
    #防止未导入信息而报错
    if di.global_class_names == 'null':
        roll_text_1.insert('insert',"\n请先导入信息，再进行数据分析！")
    else:     
        #提示模型正在评估
        roll_text_1.insert('insert',"\n数据分析中，请稍等...")
        roll_text_1.update()
        roll_text_1.insert('end',"\n光谱曲线、PCA分析结果请打开文件夹查看;\n")
        roll_text_1.insert('end',"分析完成！\n")
        ploting.main()
    return 0

#按钮事件6，数据预测
def button_event_6():
    #防止未导入信息而报错
    if di.global_class_names == 'null':
        roll_text_1.insert('insert',"\n请先导入信息，再进行标签预测！")
    else:
        #获取prediction数据的地址
        prediction.predict_txt_dir=entry_2_8.get()
        #将myGUI传给dpp的数据同样传给precision
        #直接由myGUI传比较好，直接得到模型就能预测
        #传入channel
        channel=entry_2_6.get()
        prediction.channel_s=int(channel.split(',')[0])
        prediction.channel_e=int(channel.split(',')[1])
        #传入botton_state
        #判断复选框状态，确定dpp.main()中是否全部运行
        #裁剪
        if cb_v1.get() == 0:
            prediction.botton_state[0]=0
        else:
           prediction.botton_state[0]=1   
        #均值
        if cb_v2.get() == 0:
            prediction.botton_state[1]=0
        else:
             prediction.botton_state[1]=1
        #去背景
        if cb_v3.get() == 0:
             prediction.botton_state[2]=0
        else:
             prediction.botton_state[2]=1
        #数值缩放
        if cb_v4.get() == 0:
             prediction.botton_state[3]=0
        else:
             prediction.botton_state[3]=1
        #降维
        if cb_v5.get() == 0:
             prediction.botton_state[4]=0
        else:
             prediction.botton_state[4]=1
        #生成标签
        if cb_v6.get() == 0:
             prediction.botton_state[5]=0
        else:
             prediction.botton_state[5]=1  
        #传入其他三项
        prediction.com_2_scale_state=combobox_2.get()
        prediction.com_3_dre_state=combobox_3.get()
        prediction.dred_num=int(entry_2_11.get())
        #开始预测
        roll_text_1.insert('end',"\n开始预测...")
        predict_results=prediction.main()
        for predict_result in predict_results:
                roll_text_1.insert('end',"\n  预测该条数据对应样本类型为："+str(di.global_class_names[predict_result]))
        roll_text_1.insert('end',"\n预测完成！\n")
    return 0

#按钮事件7，软件说明
def button_event_7():
    roll_text_1.insert('end',"\n操作说明：")
    roll_text_1.insert('end',"本软件用于处理XAS数据，输入采集到的txt格式的原始光谱数据，可自动进行数据前处理{剪切；均值；去背景；缩放；降维；编码}、建立机器学习模型、训练模型、评估模型、给出预测。")
    roll_text_1.insert('end',"\n  01.数据地址：输入采集到的数据所在的文件夹；")
    roll_text_1.insert('end',"\n  02.分类数量：输入样本分类数；")
    roll_text_1.insert('end',"\n  03.样本名称：输入样本对应名称；")
    roll_text_1.insert('end',"\n  04.背景区间：输入背景光谱所在区间范围，支持分段输入；")
    roll_text_1.insert('end',"\n  05.样本区间：输入样本光谱所在区间范围，支持分段输入；")
    roll_text_1.insert('end',"\n  06.剪切通道：输入要保留的光谱数据通道范围（可选操作）；")
    roll_text_1.insert('end',"\n  07.均值处理：求得背景谱的均值以及各类样本各自的均值以用于绘图；")
    roll_text_1.insert('end',"\n  08.去 背 景：可选择去平均背景还是分样本去各自样本的背景（可选操作）；")
    roll_text_1.insert('end',"\n  09.数值缩放：可选择归一化还是标准化（可选操作）；")
    roll_text_1.insert('end',"\n  10.降    维：可选择PCA、IPCA、LDA、NMF、T-SNE等降维方式（可选操作），框内输入降维后保留的维数；")
    roll_text_1.insert('end',"\n  11.生成标签：生成样本类型的标签数据；")
    roll_text_1.insert('end',"\n  12.训练算法：可选择DNN，SVM，KNN等机器学习算法，RF算法暂未开放；")
    roll_text_1.insert('end',"\n  13.数据划分：输入测试集所占比例；")
    roll_text_1.insert('end',"\n  14.评估方式：包括CM（混淆矩阵相关评价标准）、ROC、PRC、CV（交叉验证），可选择部分方式进行评估。框内输入交叉验证折数（当折数等于样本数时，为留一法）；")
    roll_text_1.insert('end',"\n  15.给定数据：输入用于预测的原始txt数据所在的文件夹，即可进行预测。\n")
    return 0

#按钮事件8，清空屏幕
def button_event_8():
    roll_text_1.delete(0.0,'end')
    #清屏会同时清掉第一行，所以重新插入
    roll_text_1.insert('end','======================控制台界面======================\n','green')
    return 0

#按钮事件9，打开文件夹
def button_event_9():
    #打开当前文件所在路径
    os.system("start explorer "+os.getcwd())
    roll_text_1.insert('end',"\n当前工作目录为："+str(os.getcwd()))
    roll_text_1.insert('end',"\n  1.file_output: 保存最终CSV文件；")
    roll_text_1.insert('end',"\n  2.file_predict:保存处理预测数据的中间CSV文件；")
    roll_text_1.insert('end',"\n  3.file_temp:   保存数据前处理中间CSV文件；")
    roll_text_1.insert('end',"\n  4.graph_output:保存模型评估、谱线分析结果图片。/n")
    return 0

#关闭程序
def button_event_10():
    root.destroy()
    return 0
    
###############################################################################
#GUI主要部分
root=Tk()
#创建主界面
root.geometry("800x600+500+200",)
root.title("机器学习算法处理XAS数据软件-by胡伟锋-2018.6.25")
#root.attributes(("-transparentcolor","blue")) 
#创建frame
frm_1=Frame(root,width=400,height=600,bg='DeepSkyBlue').place(x=0,y=0)

frm_2=Frame(root,width=400,height=170,bg='dodgerblue').place(x=400,y=0)

frm_3=Frame(root,width=400,height=190,bg='skyblue').place(x=400,y=145)
frm_4=Frame(root,width=400,height=60,bg='dodgerblue').place(x=400,y=335)

frm_5=Frame(root,width=400,height=55,bg='skyblue').place(x=400,y=395)
frm_6=Frame(root,width=400,height=40,bg='dodgerblue').place(x=400,y=435)

frm_7=Frame(root,width=400,height=40,bg='skyblue').place(x=400,y=475)

frm_8=Frame(root,width=400,height=85,bg='dodgerblue').place(x=400,y=515)
#    frm_4=Frame(root,width=400,height=100,bg='lightgreen').place(x=400,y=300)

#frame1========================================================================   
#带滚动条text，用作显示屏
roll_text_1=ScrolledText(frm_1,width=54,height=46,)
roll_text_1.insert('end','======================控制台界面======================\n','green')
roll_text_1.place(x=0,y=0)

#按钮==========================================================================    
#frame2按钮
but_2_1=Button(frm_2,text="信息导入",height=1,command=button_event_1).place(x=740,y=110)
but_2_2=Button(frm_2,text="数据处理",height=1,command=button_event_2).place(x=740,y=300)
but_2_3=Button(frm_2,text="开始训练",height=1,command=button_event_3).place(x=740,y=360)
but_2_4=Button(frm_2,text="模型评估",height=1,command=button_event_4).place(x=740,y=400)
but_2_5=Button(frm_2,text="数据分析",height=1,command=button_event_5).place(x=740,y=440)
but_2_6=Button(frm_2,text="数据预测",height=1,command=button_event_6).place(x=740,y=480)
#frame3按钮
but_2_7=Button(frm_2,text="软件说明",command=button_event_7).place(x=420,y=540)
but_2_8=Button(frm_2,text="清空屏幕",command=button_event_8).place(x=510,y=540)
but_2_9=Button(frm_2,text="打开文件",command=button_event_9).place(x=620,y=540)
but_2_10=Button(frm_2,text="退出程序",command=button_event_10).place(x=720,y=540)
   
#标签
lab_2_1=Label(frm_2,text="数据地址:",relief='groove').place(x=400,y=5)
lab_2_2=Label(frm_2,text="分类数量:",relief='groove').place(x=400,y=30)
lab_2_3=Label(frm_2,text="样本名称:",relief='groove').place(x=400,y=55)
lab_2_4=Label(frm_2,text="背景区间:",relief='groove').place(x=400,y=90)
lab_2_5=Label(frm_2,text="样本区间:",relief='groove').place(x=400,y=115)
#lab_2_6=Label(frm_2,text="预测样本:",relief='groove').place(x=400,y=440)
#lab_2_7=Label(frm_2,text="预测结果:",relief='groove').place(x=400,y=465)
lab_2_8=Label(frm_2,text="给定数据:",relief='groove').place(x=400,y=485)
#lab_2_9=Label(frm_2,text="指定列号:",relief='groove').place(x=400,y=505)
lab_2_10=Label(frm_2,text="训练算法:",relief='groove').place(x=400,y=340)
lab_2_11=Label(frm_2,text="数据划分:",relief='groove').place(x=400,y=365)
lab_2_12=Label(frm_2,text="评估方式:",relief='groove').place(x=400,y=405)

#输入框1
entry_txt_2_1=StringVar()
entry_txt_2_1.set("\\训练数据文件夹名称")
entry_2_1=Entry(frm_2,width=35,textvariable=entry_txt_2_1)
entry_2_1.place(x=480,y=5)   
#输入框2
entry_txt_2_2=StringVar()
entry_txt_2_2.set("2")
entry_2_2=Entry(frm_2,width=35,textvariable=entry_txt_2_2)
entry_2_2.place(x=480,y=30)

#a=entry_2_2.get()
#a=int(a)
#i=0
#st=''
#while(i<a):
#    i+=1
#    st+="S"+str(i)+","
    
#输入框3
entry_txt_2_3=StringVar()
entry_txt_2_3.set("S1,S2")
#entry_txt_2_3.set("S1,S2,S3,S4,S5,S6")
entry_2_3=Entry(frm_2,width=35,textvariable=entry_txt_2_3)
entry_2_3.place(x=480,y=55)
#输入框4
entry_txt_2_4=StringVar()
entry_txt_2_4.set("[1,10],[11,20]")
entry_2_4=Entry(frm_2,width=35,textvariable=entry_txt_2_4)
entry_2_4.place(x=480,y=90)
#输入框5
entry_txt_2_5=StringVar()
entry_txt_2_5.set("[21,220]")
entry_2_5=Entry(frm_2,width=35,textvariable=entry_txt_2_5)
entry_2_5.place(x=480,y=115)
#输入框6
entry_txt_2_6=StringVar()
entry_txt_2_6.set("1,200")
entry_2_6=Entry(frm_2,width=35,textvariable=entry_txt_2_6)
entry_2_6.place(x=480,y=150)
##输入框7
#entry_txt_2_7=StringVar()
#entry_txt_2_7.set("F:\DOC\Anaconda_3510\predict")
#entry_2_7=Entry(frm_2,width=35,textvariable=entry_txt_2_7)
#entry_2_7.place(x=480,y=440)
#输入框8
entry_txt_2_8=StringVar()
entry_txt_2_8.set("\\预测数据文件夹名称")
entry_2_8=Entry(frm_2,width=35,textvariable=entry_txt_2_8)
entry_2_8.place(x=480,y=485)
##输入框9
#entry_txt_2_9=StringVar()
#entry_txt_2_9.set("1,2,3,4")
#entry_2_9=Entry(frm_2,width=35,textvariable=entry_txt_2_9)
#entry_2_9.place(x=480,y=505)
#输入框10
entry_txt_2_10=StringVar()
entry_txt_2_10.set("0.2")
entry_2_10=Entry(frm_2,width=35,textvariable=entry_txt_2_10)
entry_2_10.place(x=480,y=365)
#输入框11
entry_txt_2_11=StringVar()
entry_txt_2_11.set("4")
entry_2_11=Entry(frm_2,width=15,textvariable=entry_txt_2_11)
entry_2_11.place(x=618,y=270)
#输入框12，交叉验证折数
entry_txt_2_12=StringVar()
entry_txt_2_12.set("10")
entry_2_12=Entry(frm_2,width=5,textvariable=entry_txt_2_12)
entry_2_12.place(x=690,y=407)

#复选框
#check_box=[('剪切通道',1),('均值处理',2),('去 背  景',3),('数值缩放',4),('数据降维',5),('生成标签',6)]
#y_add=0
#
#for text,value in check_box:
#    foo=IntVar()
#
#    cb=Checkbutton(root,text=text,relief='groove',variable=foo,)
#    cb.select()
#    cb.place(x=400,y=150+y_add,) 
#
#    y_add+=30
   
#全部复选框
cb_v1=IntVar()
cb_v2=IntVar()
cb_v3=IntVar()
cb_v4=IntVar()
cb_v5=IntVar()
cb_v6=IntVar()
#评估方式复选框
cb_v7=IntVar()
cb_v8=IntVar()
cb_v9=IntVar()
cb_v10=IntVar()
#复选框1
cb_2_1=Checkbutton(frm_2,text='剪切通道',relief='groove',variable=cb_v1,onvalue=1,offvalue=0)
cb_2_1.select()
cb_2_1.place(x=400,y=150)
#复选框2
cb_2_2=Checkbutton(frm_2,text='均值处理',relief='groove',variable=cb_v2,onvalue=1,offvalue=0)
cb_2_2.select()
cb_2_2.place(x=400,y=180)
#复选框3
cb_2_3=Checkbutton(frm_2,text='去 背 景 ',relief='groove',variable=cb_v3,onvalue=1,offvalue=0)
cb_2_3.select()
cb_2_3.place(x=400,y=210)
#复选框4
cb_2_4=Checkbutton(frm_2,text='数值缩放',relief='groove',variable=cb_v4,onvalue=1,offvalue=0)
cb_2_4.select()
cb_2_4.place(x=400,y=240)
#复选框5
cb_2_5=Checkbutton(frm_2,text=' 降    维 ',relief='groove',variable=cb_v5,onvalue=1,offvalue=0)
cb_2_5.select()
cb_2_5.place(x=400,y=270)
#复选框6
cb_2_6=Checkbutton(frm_2,text='生成标签',relief='groove',variable=cb_v6,onvalue=1,offvalue=0)
cb_2_6.select()
cb_2_6.place(x=400,y=300)
#评估复选框
#复选框7
cb_2_7=Checkbutton(frm_2,text='CM',relief='groove',variable=cb_v7,onvalue=1,offvalue=0)
cb_2_7.select()
cb_2_7.place(x=480,y=405)
#复选框8
cb_2_8=Checkbutton(frm_2,text='ROC',relief='groove',variable=cb_v8,onvalue=1,offvalue=0)
cb_2_8.select()
cb_2_8.place(x=530,y=405)
#复选框9
cb_2_9=Checkbutton(frm_2,text='PRC',relief='groove',variable=cb_v9,onvalue=1,offvalue=0)
cb_2_9.select()
cb_2_9.place(x=586,y=405)
#复选框10
cb_2_10=Checkbutton(frm_2,text='CV',relief='groove',variable=cb_v10,onvalue=1,offvalue=0)
cb_2_10.select()
cb_2_10.place(x=640,y=405)


#下拉框1
combobox_1=ttk.Combobox(frm_2,width=32,state='readonly')
combobox_1['values']=('总平均背景','分样本背景',)
combobox_1.current(0)
combobox_1.place(x=480,y=210)
#下拉框2
combobox_2=ttk.Combobox(frm_2,width=32,state='readonly')
combobox_2['values']=('归一化','标准化',)
combobox_2.current(0)
combobox_2.place(x=480,y=240)
#下拉框3
combobox_3=ttk.Combobox(frm_2,width=16,state='readonly')
combobox_3['values']=('PCA','IPCA','LDA','NMF','T-SNE',)
combobox_3.current(0)
combobox_3.place(x=480,y=270)
#下拉框4
combobox_4=ttk.Combobox(frm_2,width=32,state='readonly')
combobox_4['values']=('DNN','SVM','KNN','RF',)
combobox_4.current(0)
combobox_4.place(x=480,y=340)

root.mainloop()

    
        

    
    