# -*- coding: utf-8 -*-
#==============================================================================
# Author:      胡伟锋
# Created:     2019-03-14
# Version:     4.0.0
# Purpose:     SAE and 1D-CNN
#==============================================================================
"""
此模块为深度学习算法相关模块，主要包括SAE算法和CNN算法:
    ①.SAE:其功能类似于PCA，需和training.py中的其他算法配合使用
    （先运行data_preprocessing模块生成file_temp文件；
    然后再使用SAE将中间文件进一步降维后返回保存（运行dlAPI即可）；
    最后再运行training.py进行训练）。
    ②.CNN：为了最大化CNN的评估功能，已继承mlAPI中的MLFrame类，
    因此与mlAPI中的NN、SVM、kNN等模型并无使用上的区别，可在training.py中直接端到端使用；
    唯一的不同是暂无cross_validation功能，需屏蔽后再运行。
    此外，CNN为端到端的模型，因此无需先使用PCA"""
from keras.models import Model,load_model,Sequential
from keras.layers import Dense,Input,BatchNormalization,Conv1D,MaxPooling1D,Activation,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os

class SAEModel():
    def __init__(self,):
        self.train_data='file_temp/sample_4_scaled.csv'

    def load_data(self,):
        data=pd.read_csv(self.train_data,header=None,index_col=None)
        return data.T

    def model_training(self,x_train,y_train):
        #encoded
        input_layer=Input(shape=(240,),)
        encoded=Dense(128,activation='relu')(input_layer)
        encoded=BatchNormalization()(encoded)
#        encoded=Dense(32,activation='relu')(encoded)
#        encoded=BatchNormalization()(encoded)
        encoded=Dense(10,activation='sigmoid')(encoded)
        #decoded
        decoded=Dense(10,activation='relu')(encoded)
        decoded=BatchNormalization()(decoded)
#        decoded=Dense(32,activation='relu')(decoded)
#        decoded=BatchNormalization()(decoded)
        decoded=Dense(128,activation='relu')(decoded)
        decoded=BatchNormalization()(decoded)
        decoded=Dense(240,activation='sigmoid')(decoded)
        autoencoder=Model(inputs=input_layer,outputs=decoded)#完整SAE
        encoder=Model(inputs=input_layer,outputs=encoded)#编码层
        adam=Adam(lr=0.001)#训练
        autoencoder.compile(loss='mse',optimizer=adam,)
        earlystopping=EarlyStopping(monitor='loss',patience=20,verbose=2)
        callbacks=[earlystopping,]
        #training,epochs:迭代次数
        autoencoder.fit(x_train,y_train,batch_size=32,epochs=500,callbacks=callbacks)
        #Draw the network structure diagram
        plot_model(model=autoencoder,to_file='AE_we_model.png',show_shapes=True)
#        plot_model(model=encoder,to_file='E_we_model.png',show_shapes=True)
        #Save model
        autoencoder.save("model/SAE.model")
        encoder.save("model/SAEencoder.model")
        return 0
        
    def predicting(self,data):
        #载入训练好的转换模型
        model=load_model("Model/SAEencoder.model",)
        res=model.predict(data)
        res=pd.DataFrame(res)
        #保存数据,无损替换mlAPI中PCA生成的降维数据，使工作量最小
        res.to_csv("file_temp/sample_8_dre.csv",header=None,index=None)
        res.to_csv("file_output/sample.csv",header=None,index=None)
        print('完成SAE降维，请到training.py模块中继续训练SAE-DNN！')
        return 0

from mlAPI import MLFrame
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer
from sklearn.model_selection import cross_val_score,KFold,RepeatedKFold
from sklearn.metrics import accuracy_score
class CNNModel(MLFrame):
    def __init__(self,):
        super(CNNModel,self).__init__()
        self.train_data='file_temp/sample_4_scaled.csv'
        self.label='file_output/label.csv'
        
    def load_xy(self,):
        data=pd.read_csv(self.train_data,header=None,index_col=None)
        label=pd.read_csv(self.label,header=None,index_col=None)
        data=np.asarray(data.T)
        data=np.expand_dims(data,axis=2)
        print(data.shape)
        return data,label.T
    
    def reset_x(self,):#针对CNN模型对输入Xy数据格式进行调整
        def reset_data(data):
#            data=np.asarray(data.T)
            data=np.expand_dims(data,axis=2)
            return data
        #扩展X的维度，供1D-CNN使用
        self.x_train=reset_data(self.x_train)
        self.x_test=reset_data(self.x_test)
        #保存转换前y的编码
        self.y_test_temp=self.y_test
        self.y_train_temp=self.y_train
        #对y重新进行转换，转换为LabelBinarizer格式
        lb=LabelBinarizer()
        self.y_train=lb.fit_transform(self.y_train)
        self.y_test=lb.transform(self.y_test)
        print('done!')
        return self.x_train,self.x_test,self.y_train,self.x_test
        
    #仿照mlAPI中的classifiers格式对CNN进行封装，使具有相同的输入输出功能
    def classifiers(self,):
        self.reset_x()#转换数据格式
        #搭建CNN模型
        model=Sequential()
        #Block-1
        model.add(Conv1D(filters=32,kernel_size=5,activation='relu',input_shape=(240,1)))
        model.add(Conv1D(filters=32,kernel_size=5,activation='relu'))
        model.add(MaxPooling1D(pool_size=5,))
        #Block-2
        model.add(Conv1D(filters=64,kernel_size=5,activation='relu'))
        model.add(Conv1D(filters=64,kernel_size=5,activation='relu')) 
        model.add(MaxPooling1D(pool_size=5,strides=1))
        #Block-2
        model.add(Conv1D(filters=128,kernel_size=5,activation='relu'))
        model.add(Conv1D(filters=128,kernel_size=5,activation='relu')) 
        model.add(MaxPooling1D(pool_size=3,strides=1))
        #Block-3
        model.add(Conv1D(filters=256,kernel_size=3,activation='relu'))
        model.add(Conv1D(filters=256,kernel_size=3,activation='relu')) 
        model.add(GlobalAveragePooling1D())
        #Block-4
#        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(5,activation='softmax'))
        model.summary()#打印模型参数概况
        #loss_function
        adam=Adam(0.0005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,metrics=['mse','acc'])
        #early_stopping
        earlystopping=EarlyStopping(monitor='loss',patience=30,verbose=2)
        callbacks=[earlystopping,]
        model.fit(self.x_train,self.y_train,
                  batch_size=16,epochs=20,shuffle=True,callbacks=callbacks)
        #complete model：
#        plot_model(model=model,to_file='model/1DCNN.png',show_shapes=True)
#        model.save("model/1DCNN_epoch300.model")
#        self.y_predict=model.predict(self.x_test)
        #仿照classifiers进行四大参数输出
        self.y_predict=model.predict_classes(self.x_test)#1
        self.y_predict_proba=model.predict_proba(self.x_test)#2
        #计算self.train_accuracy
        self.y_train=self.y_train_temp
        train_pre=model.predict_classes(self.x_train)
        self.train_accuracy=accuracy_score(train_pre,self.y_train)#3
        #计算self.test_accuracy
        self.y_test=self.y_test_temp
        self.test_accuracy=accuracy_score(self.y_predict,self.y_test)
        return model
        
if __name__=='__main__':
    #SAE
    sm=SAEModel()
#    x_train=sm.load_data()
#    sm.model_training(x_train=x_train,y_train=x_train)
#    sm.predicting(x_train)
    
    #CNN
    cnn=CNNModel()
#    cnn.training()
#    cnn.evaluation(K_Fold=3)
    
        