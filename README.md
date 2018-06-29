### Project1.DNNProcessingXAS1.0.0
    ————用深度神经网络算法处理X射线吸收光谱数据（XAS），对被测物体进行分类识别。
### Project2.DNNProcessingXAS2.1.0
    ————对1.0.0版本代码进行重构，增强其可扩展性；增加SVM、KNN等算法；使数据输入及接收更智能。（总代码量>1600行）
### Project3.MLProcessingXASwithGUI3.0.0
    ————程序包含以下功能：
            ①.将txt批量转成csv、进行剪切、分割、重排、均值；
            ②.去背景、归一化、标准化、数据降维{PCA，IPCA，LDA，MNF，t-SNE}；
            ③.用DNN、SVM、KNN等机器学习模型进行训练；
            ④.用混淆矩阵、MCC系数、Cohen-Kappa系数、F1值、准确率、召回率、ROC、AUC、PRC、K折交叉验证等评估指标进行评估；
            ⑤.对曲线进行分析、对PCA结果进行分析；
            ⑥.预测给定txt文件的分类类别。
    ————用TKinter将代码封装成GUI，并用pyinstaller生成exe文件。
