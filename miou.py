import json
import os
import numpy as np
from pycocotools.mask import *


def read_json(filename):
    with open(filename,'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

test_json = read_json("/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/val.json")
image_id_name = {}
testimage_meta = {}
for i in range(len(test_json["images"])):
    image_id_name[test_json["images"][i]["id"]] = test_json["images"][i]["file_name"]
    testimage_meta[test_json["images"][i]["file_name"]] = [test_json["images"][i]["height"],test_json["images"][i]["width"]]
    
# a list
mmd_json = read_json("/home/wfy/code/mmdetection-master-old/mask_100_au_cosine_cas/results_val_720.pkl.json")
# Required information：image_name; score(therehold？); category_id; segmentation的size和counts用于decode
# generate a dict，{ key为image_id,value为{1:npy1,2:npy2}... }
# The "OR" operation for each numpy
#test = decode(mmd_json[0]['segmentation'])
result_dict = {}
for key in testimage_meta:
    temp = np.zeros(testimage_meta[key])>0
    category_dict = {
        1:temp,
        2:temp,
        3:temp,
        4:temp,
        5:temp}
    result_dict[key] = category_dict
    
#therehold = [0.4,0.35,0.4,0.4,0.2]
#therehold = [0.2,0.08,0.15,0.2,0.1]
#therehold = [0.5,0.5,0.5,0.5,0.5]
therehold = [0.3,0.3,0.3,0.3,0.3]
#therehold = [0,0,0,0,0]
for i in range(len(mmd_json)):
    image_name = image_id_name[mmd_json[i]['image_id']]
    score = mmd_json[i]['score']
    category_id = mmd_json[i]['category_id']
    segmentation = mmd_json[i]['segmentation']
    if category_id == 1:
        if score >= therehold[0]:
            npy = decode(segmentation)>0
            result_dict[image_name][1] = result_dict[image_name][1] + npy
    elif category_id == 2:
        if score >= therehold[1]:
            npy = decode(segmentation)>0
            result_dict[image_name][2] = result_dict[image_name][2] + npy
    elif category_id == 3:
        if score >= therehold[2]:
            npy = decode(segmentation)>0
            result_dict[image_name][3] = result_dict[image_name][3] + npy
    elif category_id == 4:
        if score >= therehold[3]:
            npy = decode(segmentation)>0
            result_dict[image_name][4] = result_dict[image_name][4] + npy
    else:
        if score >= therehold[4]:
            npy = decode(segmentation)>0
            result_dict[image_name][5] = result_dict[image_name][5] + npy
            
truth_dic=dict()
for key in testimage_meta:
    temp = np.zeros(testimage_meta[key])>0
    category_dict = {
        1:temp,
        2:temp,
        3:temp,
        4:temp,
        5:temp}
    truth_dic[key] = category_dict

def seg2npy(h,w,seg):
    rles = frPyObjects(seg, h, w)
    rle = merge(rles)    
    return decode(rle)

for i in range(len(test_json['annotations'])):
    image_name = image_id_name[test_json['annotations'][i]['image_id']]
    #score = mmd_json[i]['score']
    category_id = test_json['annotations'][i]['category_id']
    segmentation = test_json['annotations'][i]['segmentation']
    h=testimage_meta[ image_name ][0]
    w=testimage_meta[ image_name ][1]
    if category_id == 1:
        npy = seg2npy(h,w,segmentation)>0
        truth_dic[image_name][1] = truth_dic[image_name][1]+npy
    elif category_id == 2:
        npy = seg2npy(h,w,segmentation)>0
        truth_dic[image_name][2] = truth_dic[image_name][2]+npy
    elif category_id == 3:
        npy = seg2npy(h,w,segmentation)>0
        truth_dic[image_name][3] = truth_dic[image_name][3]+npy
    elif category_id == 4:
        npy = seg2npy(h,w,segmentation)>0
        truth_dic[image_name][4] = truth_dic[image_name][4]+npy
    else:
        npy = seg2npy(h,w,segmentation)>0
        truth_dic[image_name][5] = truth_dic[image_name][5]+npy
        
iou_dic=dict()
for i in range(1,6):
    iou_dic[i]=[0,0]

for file_name in result_dict.keys():
    for class_ in range(1,6):
        truth_label=truth_dic[file_name][class_]
        result_label=result_dict[file_name][class_]
        truth_region=set(np.where(truth_label.reshape(-1)!=0 )[0].tolist())
        result_region=set(np.where(result_label.reshape(-1)!=0)[0].tolist())
        i_num=len(truth_region& result_region)
        u_num=len(truth_region|  result_region)
        if u_num!=0:
            iou_dic[class_][0]+=i_num
            iou_dic[class_][1]+=u_num
mean = 0
for i in range(5):
    mean = iou_dic[i+1][0]/iou_dic[i+1][1] + mean
    print(iou_dic[i+1][0]/iou_dic[i+1][1])
print(iou_dic)
print(mean/5)



