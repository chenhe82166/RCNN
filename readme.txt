17flowers��17�ִ�����ǩ��ѵ�����ݼ�
2flowers��2�ִ�����ǩ�������ǩ��ѵ�����ݼ�
train_alexnet.py����ʼ����
preprocessing_RCNN.py��IOU�㷨
fine_tune_RCNN.py��΢������
RCNN_output.py����� ���

ʵ�����̣�
1.����train_alexnet.py�ļ������ļ�����ȡtrain_list.txt�ļ�������17flowersͼƬ����ѵ����
����model_save.model��model_save.model.meta��output�ļ��м�dataset.pkl�ļ���
model_save.model�Ǳ��������ģ�ͣ�output�ǿ��ӻ��ļ���dataset.pkl�ļ�ΪͼƬ�����ı����ļ���

2.����fine_tune_RCNN.py�ļ���������ȡrefine_list.txt�ļ�������2flowers�ļ��е�ͼƬ����΢��ѵ����
������preprocessing_RCNN.py�ļ��������ǩ��ƥ��ѵ����
����fine_tune_model_save.model��fine_tune_model_save.model.meta��output_RCNN��dataset.pkl�ļ���
���ɵ�����ģ���ļ��Ƿ������ֱ�ӵ��ã�

3.����RCNN_output.py�ļ������ļ����������һ���ΪSVM�����࣬����ӦС��ģ���ݵ�ѵ����֮��
����һ��ͼƬ��ͨ�������ѡ�㷨����500����Ȼ��ɸѡ��һЩ��Ч�Ŀ����뵽���磬ͨ��SVM�����࣬��ʣ��
һЩ���������Ŀ򣬽����÷Ǽ���ֵ������ɸѡ�����·��������Ŀ�����ͨ��IOUȡ���Ŀ�
���ڵ�NMS�㷨���Է���������жϣ���������SVM����û�и��ʣ�������ͼƬ����ͼƬ�ϻ���