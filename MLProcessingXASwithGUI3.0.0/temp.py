# -*- coding: utf-8 -*-
#import ploting
import training
import data_preprocessing
import tkinter as tk
root=tk.Tk()
#side可选为top,bottom,left,right
#设置主窗口大小
#root.geometry("800x600+10+10")
#主窗口名称
root.title("机器学习算法处理XAS数据软件")

#标签
#label=tk.Label(root,text="hello world!",height=2,width=8)
#label.pack()

#创建按钮
#tk.Button(root,text="确定",height=2,width=10,command=False).pack(side='left')
#tk.Button(root,text="取消",height=2,width=10).pack(side='left')
#tk.Button(root,text="警告",height=2,width=10,).pack(side='right')
#tk.Button(root,text="退出",height=2,width=10,command=root.quit).pack(side='right')


#创建输入框
#f1=tk.Frame(root)
#tk.Label(f1,text="输入数据：").pack(side='left',padx=5,pady=10)
#e1=tk.StringVar()
#tk.Entry(f1,width=50,textvariable=e1).pack(side='left')
#e1.set('F:\DOC\Anaconda_3510\02.Practice\P04.CodeRefactoring180612')
#f1.pack()

#创建复选框
#check_box=[('数据处理',1),('训练',2),('绘图',3),('演示',4)]
#for text,value in check_box:
#    foo=tk.IntVar()
#    c=tk.Checkbutton(root,text=text,variable=foo,)
#    c.pack(anchor='w',)

#创建消息
tk.Message(root,text="本程序用于处理X射线吸收光谱(XAS)数据",width=500,relief='groove',bg='lightblue').pack(side='top',padx=10,pady=10)


#正式创建frame
frm_1=tk.Frame(root,width=300,height=500,bg='lightblue')
frm_2=tk.Frame(root,width=300,height=500,bg='lightblue')
frm_3=tk.Frame(root,width=300,height=500,bg='lightblue')
frm_4=tk.Frame(root,width=300,height=500,bg='lightblue')
frm_5=tk.Frame(root,width=1240,height=40,bg='lightblue')

#创建按钮响应
def but_click_event_1():
    data_preprocessing.main()
    return 0
def but_click_event_2():
#    root.withdraw()
    return 0
    


#1.创建控件
#label_1=tk.Label(frm_1,text="数据地址：").pack(side='top',padx=0,pady=10)
label_1=tk.Label(frm_1,text="数据地址：").place(x=10,y=10)
label_1.pack(side='top',padx=0,pady=10)
#label_1.place(x=10,y=10)
e_1=tk.Entry(frm_1,width=20,).pack(side='top')
but_1=tk.Button(frm_1,text="开始处理",width=8,command=but_click_event_1).pack(side='left')
but_1=tk.Button(frm_1,text="取消",width=8,command=but_click_event_2).pack(side='left')

#3.控件
#4.控件

#frm_3=tk.Frame(width=500,height=30,bg='white')
#frm_4=tk.Frame(width=200,height=500,bg='white')
#布局
#frm_1.grid(row=0,column=0,columnspan=2,padx=1,pady=3 )
#frm_2.grid(row=1,column=0,columnspan=2,padx=1,pady=3 )
#frm_3.grid(row=2,column=0,columnspan=2,padx=1,pady=3 )
#frm_4.grid(row=0,column=2,columnspan=3,padx=1,pady=3 )
##固定大小
#frm_1.grid_propagate(0)
#frm_2.grid_propagate(0)
#frm_3.grid_propagate(0)
#frm_4.grid_propagate(0)
frm_5.pack(side='bottom',padx=1,pady=10)
frm_1.pack(side='left',padx=10,pady=10)
frm_2.pack(side='left',padx=10,pady=10)
frm_3.pack(side='left',padx=10,pady=10)
frm_4.pack(side='left',padx=10,pady=10)

#frm_3.pack()
#frm_4.pack()

#进入主循环
root.mainloop()
