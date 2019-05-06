import json
import os
import numpy as np
from pycocotools.mask import *


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict
def seg2npy(h,w,seg):
    rles = frPyObjects(seg, h, w)
    rle = merge(rles)
    return decode(rle)
def trans_segmentation(test_json,json_dict,therehold,result_dict=None):
    image_id_name = {}
    testimage_meta = {}
    for i in range(len(test_json["images"])):
        image_id_name[test_json["images"][i]["id"]] = test_json["images"][i]["file_name"]
        testimage_meta[test_json["images"][i]["file_name"]] = [test_json["images"][i]["height"],
                                                               test_json["images"][i]["width"]]
    if result_dict==None:
        result_dict = {}
        for key in testimage_meta:
            temp = np.zeros(testimage_meta[key])>0
            category_dict = {
                1: temp,
                2: temp,
                3: temp,
                4: temp,
                5: temp}
            result_dict[key] = category_dict
    for i in range(len(json_dict)):
        print(i)

        image_name = image_id_name[json_dict[i]['image_id']]
        score = json_dict[i]['score']
        category_id = json_dict[i]['category_id']
        segmentation = json_dict[i]['segmentation']
        if category_id == 1:
            if score >= therehold[0]:
                npy = decode(segmentation)*score
                result_dict[image_name][1] = result_dict[image_name][1] + npy
        elif category_id == 2:
            if score >= therehold[1]:
                npy = decode(segmentation)*score
                result_dict[image_name][2] = result_dict[image_name][2] + npy
        elif category_id == 3:
            if score >= therehold[2]:
                npy = decode(segmentation)*score
                result_dict[image_name][3] = result_dict[image_name][3] + npy
        elif category_id == 4:
            if score >= therehold[3]:
                npy = decode(segmentation)*score
                result_dict[image_name][4] = result_dict[image_name][4] + npy
        else:
            if score >= therehold[4]:
                npy = decode(segmentation)*score
                result_dict[image_name][5] = result_dict[image_name][5] + npy
    return result_dict
def true_segmentation(test_json):

    image_id_name = {}
    testimage_meta = {}
    for i in range(len(test_json["images"])):
        image_id_name[test_json["images"][i]["id"]] = test_json["images"][i]["file_name"]
        testimage_meta[test_json["images"][i]["file_name"]] = [test_json["images"][i]["height"],
                                                               test_json["images"][i]["width"]]
    truth_dic = dict()
    for key in testimage_meta:
        temp = np.zeros(testimage_meta[key]) > 0
        category_dict = {
            1: temp,
            2: temp,
            3: temp,
            4: temp,
            5: temp}
        truth_dic[key] = category_dict
    for i in range(len(test_json['annotations'])):
        image_name = image_id_name[test_json['annotations'][i]['image_id']]
        # score = mmd_json[i]['score']
        category_id = test_json['annotations'][i]['category_id']
        segmentation = test_json['annotations'][i]['segmentation']
        h = testimage_meta[image_name][0]
        w = testimage_meta[image_name][1]
        if category_id == 1:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][1] = truth_dic[image_name][1] + npy
        elif category_id == 2:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][2] = truth_dic[image_name][2] + npy
        elif category_id == 3:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][3] = truth_dic[image_name][3] + npy
        elif category_id == 4:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][4] = truth_dic[image_name][4] + npy
        else:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][5] = truth_dic[image_name][5] + npy
    return truth_dic
def threshold_result(result_dict,threshold):
    for img_indx,category_result in result_dict.items():
        for category_id,confidence in category_result.items():
            if category_id==1:
                confidence[confidence<=threshold[0]] = 0
                confidence[confidence>threshold[0]] = 1
            if category_id==2:
                confidence[confidence <= threshold[1]] = 0
                confidence[confidence > threshold[1]] = 1
            if category_id==3:
                confidence[confidence <= threshold[2]] = 0
                confidence[confidence > threshold[2]] = 1
            if category_id==4:
                confidence[confidence <= threshold[3]] = 0
                confidence[confidence > threshold[3]] = 1
            if category_id==5:
                confidence[confidence <= threshold[4]] = 0
                confidence[confidence > threshold[4]] = 1
    return result_dict
if __name__=="__main__":
    test_json = read_json("/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/val.json")


    # a list
    mmd_json_1 = read_json("/home/wfy/code/mmdetection-master-old/mask_100_au_cosine_cas/results_val_720.pkl.json")
    mmd_json_2 = read_json("/home/wfy/code/mmdetection-master-old/mask_100_au_cosine_cas/results_val_880.pkl.json")
    mmd_json_3 = read_json("/home/wfy/code/mmdetection-master-old/mask_100_au_cosine_cas/results_val_1040.pkl.json")
    mmd_json_4 = read_json("/home/wfy/code/mmdetection-master-old/mask_100_au_cosine_cas/results_val_1200.pkl.json")
    all_json=[mmd_json_1,mmd_json_2,mmd_json_3,mmd_json_4]
    therehold = [0, 0, 0, 0, 0]
    #therehold = [0.3, 0.3, 0.3, 0.3, 0.3]
    #therehold = [0.3, 0.3, 0.3, 0.3, 0.3]
    result_dict = None
    for mmd_json in all_json:

        result=trans_segmentation(test_json,mmd_json,therehold,result_dict)
        result_dict=result
    total_therehold = [1.3, 1.3, 1.3, 1.3, 1.3]
    result_dict = threshold_result(result_dict, total_therehold)
    truth_dict=true_segmentation(test_json)
    iou_dic = dict()
    for i in range(1, 6):
        iou_dic[i] = [0, 0]

    for file_name in result_dict.keys():
        for class_ in range(1, 6):
            truth_label = truth_dict[file_name][class_]
            result_label = result_dict[file_name][class_]
            truth_region = set(np.where(truth_label.reshape(-1) != 0)[0].tolist())
            result_region = set(np.where(result_label.reshape(-1) != 0)[0].tolist())
            i_num = len(truth_region & result_region)
            u_num = len(truth_region | result_region)
            if u_num != 0:
                iou_dic[class_][0] += i_num
                iou_dic[class_][1] += u_num
    mean = 0
    for i in range(5):
        mean = iou_dic[i + 1][0] / iou_dic[i + 1][1] + mean
        print(iou_dic[i + 1][0] / iou_dic[i + 1][1])
    print(iou_dic)
    print(mean / 5)