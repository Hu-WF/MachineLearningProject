## 1.初步完成
### 1.该项目的核心功能基于python中sklearn、pandas、numpy、scipy、matplotlib等机器学习或数据分析API编写，API对应网址：
	①.http://scikit-learn.org/stable/
	②.http://pandas.pydata.org/
	③.http://www.numpy.org/
	④.https://www.scipy.org/
	⑤.https://matplotlib.org/

# 2.项目包含以下四个模块：
	①.MLFunctions.py：程序的所有类和函数存放于此处，集中管理和调用；
	②.MLDataProcessing.py：针对不同类型的实验，选择性调用数据处理程序；
	③.MLTraining.py：控制模型进行训练和评估；
	④.MLPloting.py：绘制数据图表。
	运行次序为①②③④或①②④③。

## 3.主要实现以下六大功能：
	①.数据读入整合(生成文件夹，txt批量转csv，合并多个csv，重排单个csv）；
	②.数据前处理（csv分割，裁剪，均值化，归一化，去背景）；
	③.数据降维(PCA降维，可扩展LDA、SVD等其他降维方式)；
	④.训练模型（神经网络训练算法，可扩展SVM、K-neighbor、RF等算法）；
	⑤.模型评估（准确率、多元混淆矩阵、多元评估报告、kappa系数、MCC系数、ROC曲线、AUC值）；
	⑥.绘图（背景图X1，光谱图X3，降维图X3）。
	程序可扩展性强，后续可进行大量的横向、纵向扩展。

## 4.2018.5.8更新：针对癌症实验数据的特性，及数据处理的要求，做出了如下改进：
	①.增加了生成线性插值的背景的函数，扩展了deback方法，使其能够去插值背景；
	②.增加了0-1归一化方式，重构了正则化方法，可选执行标准化还是归一化；
	③.增加了MLDemo.py，用于分析数据时使用，可在里面灵活画图观察；
	④.Draw()类里面增加了draw_specific_col（）函数，可根据要求画出指定文件的指定列数据，可同时在一图里画多列，cols指定第几列，	   可接受无数多个列。便于调试观察。
	⑤.新增了NMF、T-SNE和LDA三种降维方式。

# 5.2018.5.9更新：
	①.评估方式新增了K折交叉验证；

# 6.2018.5.11更新：
	①.新增了MLCreateCTData.py，用于通过去背景、标准化、降维等一系列处理，生成CT_x_data.csv用于CT预测；
	②.新增了ttc.combine_csv()函数，用于将A.csv和B.csv按列合并成AB.csv；
	③新增了按行分离数据的函数；
	④.Matlab中新增了了小波降噪函数。

# 7.2018.5.14全面更新：
	为塑料分类问题和癌症样本分类问题添加了许多新方法：
	01.新增SampleInformation()类，传递样本数，样本名等基础信息；
	02.新增combine_csv(self,csv_A,csv_B,csv_AB):
	03.新增split_data_by_row(self,file_to_split,start_row,end_row,save_name):
	04.新增linear_interpolation(self,start_csv,end_csv,point_num,save_name):
	05.新增create_interpolation_back(self,start_back1,end_back1,start_back2,end_back2,rotation_sampling=False):
	06.新增universal_deback_mean(self,sample_data,back_data,debacked_data):
	07.新增data_normalization(self,filename,save_name,scale_to_01=False):
	08.标准化新增归一化选项scale(self,scaled_to_01=False):
	09.DReduction()类新增universal_pca(self,csv_name,pcaed_name):Nmf(self,):Tsne(self,):Lda(self):几种降维算法
	10.新增draw_PR_curve(self,):用于绘制PR曲线；
	11.新增CT_predict(self,):用于对ct重建谱进行预测；
	12.新增SVM(MLFrame)类；
	13.新增KNeighbors(MLFrame)类:
	14.新增draw_specific_col(self,csv_name,save_name="test",*cols):可画出指定csv的指定列的曲线。用于观察分析；
	15.新增MLCreateNoiseData.py：用于通过给样本加噪声创建新样本；
	16.新增MLCreateCTData.py   ：用于同步去背景、标准化、降维等操作，生成CT_x_data.csv用于预测；
	17.新增MLDemo.py	   ：用于绘制观察分析的曲线；
	18.新增temp.py/test.py
	19.新增文件夹file_ct：用于存储生成CT_x_data.csv的过程文件；
	20.新增graph_demo：用于存储观察分析的曲线；
	21.新增noise：用于存储生成添加噪声样本的过程文件；
	22.新增testing：用于尝试替换ct预测以达到100%准确率的过程文件
