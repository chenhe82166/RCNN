17flowers：17种带类别标签的训练数据集
2flowers：2种带类别标签及区域标签的训练数据集
train_alexnet.py：初始网络
preprocessing_RCNN.py：IOU算法
fine_tune_RCNN.py：微调网络
RCNN_output.py：输出 检测

实验流程：
1.运行train_alexnet.py文件，该文件将读取train_list.txt文件，读入17flowers图片进行训练，
生成model_save.model、model_save.model.meta、output文件夹及dataset.pkl文件，
model_save.model是保存的网络模型，output是可视化文件，dataset.pkl文件为图片处理后的保存文件；

2.运行fine_tune_RCNN.py文件，它将读取refine_list.txt文件，载入2flowers文件中的图片进行微调训练，
并调用preprocessing_RCNN.py文件对区域标签做匹配训练，
生成fine_tune_model_save.model、fine_tune_model_save.model.meta、output_RCNN及dataset.pkl文件，
生成的网络模型文件是方便后期直接调用；

3.运行RCNN_output.py文件，该文件将网络最后一层改为SVM来分类，以适应小规模数据的训练，之后，
输入一张图片，通过区域候选算法产生500个框，然后筛选出一些有效的框输入到网络，通过SVM做分类，会剩下
一些符合条件的框，接着用非极大值抑制做筛选，留下符合条件的框，这里通过IOU取最大的框，
正宗的NMS算法是以分类概率来判断，但这里用SVM所以没有概率，最后输出图片，在图片上画框。