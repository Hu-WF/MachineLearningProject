### 一.2018.6.20更新程序2.0.0版：
    ①.重构代码以增强程序可拓展性，进一步抽象封装；
    ②.使data_processing中的子函数变成并行执行（可选）。

### 二.重构后程序组成：
    ①.mlAPI：所有顶层类和函数封装，不涉及具体过程；
    ②.mlFunctions：次一层类和函数，包括针对具体数据的处理过程；
    ③.data_information：数据包含的原始信息num和names；文件存储名；
    ④.data_processing：从txt到生成back、sample、label四大模块的操作（并行操作）；
    ⑤.training：训练和评估过程；
    ⑥.ploting：根据训练结果作图。
