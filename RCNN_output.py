from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
from selectivesearch import selective_search
from PIL import Image
import matplotlib.pyplot as plt#matplotlib是一个绘图库
import matplotlib.patches as mpatches
import os.path
import skimage#skimage.io.imread读出图片数据为numpy格式
from sklearn import svm#sklearn中包含fit（），predict（）等方法，只要输入训练样本和标记及模型的一些参数，自然就直接得出分类的结果
#SVM可用来分类 SVC，也可用来预测回归 SVR
import preprocessing_RCNN as prep
import os

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Load testing images
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

def image_proposal(img_path):
    img = skimage.io.imread(img_path)
    img_lbl, regions = selective_search.selective_search(img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue
        #resize to 224 * 224 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        #Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if ((x == 0 or y == 0) and (w > 200 or h > 200)) or w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        im = Image.fromarray(proposal_img)
        resized_proposal_img = resize_image(im, 224, 224)
        candidates.add(r['rect'])
        img_float = pil_to_nparray(resized_proposal_img)
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices

# Load training images
def generate_single_svm_train(one_class_train_file):
    trainfile = one_class_train_file
    savepath = one_class_train_file.replace('txt', 'pkl')
    images = []
    Y = []
    if os.path.isfile(savepath):
        print("restoring svm dataset " + savepath)
        images, Y = prep.load_from_pkl(savepath)
    else:
        print("loading svm dataset " + savepath)
        images, Y = prep.load_train_proposals(trainfile, 2, threshold=0.3, svm=True, save=True, save_path=savepath)
    return images, Y
    
# Use a already trained alexnet with the last layer redesigned
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    #network = local_response_normalization(network)
    #network = conv_2d(network, 384, 3, activation='relu')
    #network = conv_2d(network, 384, 3, activation='relu')
    #network = conv_2d(network, 256, 3, activation='relu')
    #network = max_pool_2d(network, 3, strides=2)
    #network = local_response_normalization(network)
    #network = fully_connected(network, 512, activation='tanh')
    #network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='tanh')
    #network = dropout(network, 0.5)
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

# Construct cascade svms

def train_svms(train_file_folder, model):
    listings = os.listdir(train_file_folder)
    svms = []
    for train_file in listings:
        if "pkl" in train_file:
            continue
        X, Y = generate_single_svm_train(train_file_folder+train_file)
        train_features = []
        for i in X:
            feats = model.predict([i])
            train_features.append(feats[0])
        print("feature dimension")
        print(np.shape(train_features))
        clf = svm.LinearSVC()
        print("fit svm")
        clf.fit(train_features, Y)#train the svc model
        svms.append(clf)
    return svms



def non_max_suppression_slow(boxes, overlapThresh):
    
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 -x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area) 


    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    suppress = [last]
    
    for pos in range(0, last):
        
        j = idxs[pos]
        xx1 = x1[j]
        yy1 = y1[j]
        xx2 = x2[j]
        yy2 = y2[j]

        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)
        
        overlap = float(w * h) / area[i]

        if overlap > overlapThresh:
                suppress.append(pos)
            
    idxs = np.delete(idxs, suppress)
    return boxes[pick]


if __name__ == '__main__':
    train_file_folder = 'svm_train/'
    img_path = 'test-images/test9.jpg'
    imgs, verts = image_proposal(img_path)
    net = create_alexnet(3)
    model = tflearn.DNN(net)#define model
    model.load('fine_tune_model_save.model')
    svms = train_svms(train_file_folder, model)
    print("Done fitting svms")
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))
    results = []
    results_label = []
    count = 0
    for f in features:
        for i in svms:
            pred = i.predict(f)
            print(pred)
            if pred[0] != 0:
                results.append(verts[count])
                results_label.append(pred[0])
        count += 1

    print("result:")
    print(results)


    boxhaha = np.array(results)
    print(boxhaha)#打印所有的候选框
    pick = non_max_suppression_slow(boxhaha, 0.3)
    print(pick)#打印非极大值抑制所选的框
    
    for i in range(0, len(results)):
        print(i,boxhaha[i], pick[0])#按顺序打印出所有的候选框及对比的非极大值抑制框
        if boxhaha[i].all() == pick[0].all():#找到与非极大值抑制的框位置一样的框
            asd_number = i
        else:
            continue
    print(asd_number)
    
    print("result label:")
    print(results_label)#打印出所有框的类别
    asd = results_label[asd_number-1]#最终框的类别
    
    
    #max_results_label = {}
    #for i in results_label:
        #if i not in max_results_label:
            #max_results_label[i] = 1
        #else:
            #max_results_label[i] += 1
    #l_results_label = [(key,max_results_label[key]) for key in max_results_label.keys()]
    #print("last_result:")
    #asd = max(l_results_label, key = lambda x: x[1])[0]
    #print("该花为第%d类花卉" % asd)
    
    img = skimage.io.imread(img_path)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)#用来绘制二维图
    for x, y, w, h in pick:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        if asd == 1:
            flow_class = "First class flower"
        else:
            flow_class = "Second class flower"
        ax.text(x, y+200, format(flow_class), color='white', fontsize=20)#用来在绘制的图上加字
        ax.add_patch(rect)

    plt.show()#显示出创建的所有绘图对象

#将区域标签与类别概率添加到一个数组中，形如[(1,2,3,4,5),()],然后调用nms.py
#true_boxes1 = nms_max(boxes_nms, overlapThresh=0.3)
#true_boxes = nms_average(np.array(true_boxes1), overlapThresh=0.07)









