# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import json
from pycocotools.mask import *
from pycocotools import mask as maskUtils

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict
def write_json(filename, file_dict):
    with open(filename, "w") as f:
        json.dump(file_dict, f)
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
def show_segmentation(image_name,preds,segmentation_dir):
    import cv2
    height,width=(preds.shape[1], preds.shape[2]) # (height,width)
    segmentation=np.zeros((height,width,3))
    colormap=[(127, 20, 22), (9, 128, 64), (127, 128, 51), (40, 41, 115), (125, 39, 125)]
    for cls_id in range(len(preds)):
        mask=preds[cls_id][:,:]
        segmentation[:,:,0]= segmentation[:,:,0]*(1-mask)+colormap[cls_id][0] * mask
        segmentation[:,:,1] = segmentation[:,:,1]*(1-mask)+colormap[cls_id][1] * mask
        segmentation[:,:,2] = segmentation[:,:,2]*(1-mask)+colormap[cls_id][2] * mask
    print(os.path.join(segmentation_dir,image_name))
    cv2.imwrite(os.path.join(segmentation_dir,image_name),segmentation)


# json_dir="/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/val.json"
# segmentation_val_dir = "/home/wfy/code/jinnan_2/val_segmentation"
# mkdir(segmentation_val_dir)
# npz_dic={}
# with open(json_dir,'r') as load_f:
#     load_dict=json.load(load_f)
# for i in range(len(load_dict['images'])):
#     image_id=load_dict['images'][i]['id']
#     height=load_dict['images'][i]['height']
#     width=load_dict['images'][i]['width']
#     file_name=load_dict['images'][i]['file_name']
#     npz_dic[image_id]=[np.zeros((height,width),dtype=bool)]*5
# for i in range(len(load_dict['annotations'])):
#     category_id=load_dict['annotations'][i]['category_id']
#     image_id=load_dict['annotations'][i]['image_id']
#     segmentation=load_dict['annotations'][i]['segmentation']
#     height=npz_dic[image_id][0].shape[0]
#     width=npz_dic[image_id][0].shape[1]
#     fortran_mask = decode(maskUtils.frPyObjects(segmentation, height, width))[:,:,0]
#     npz_dic[image_id][category_id-1]=npz_dic[image_id][category_id-1]+fortran_mask
# for index_img,npz_list in npz_dic.items():
#     file_name=str(index_img)+'.jpg'
#     preds = []
#     for i in range(len(npz_list)):
#         npz_list[i]=npz_list[i].astype(np.uint8)
#         npz_list[i][npz_list[i]>=1]=1
#         preds.append(npz_list[i])
#     preds_np = np.array(preds)  # fake prediction
#     show_segmentation(file_name, preds_np, segmentation_val_dir)



val_dict = read_json("/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/val.json")
load_dict = read_json("/home/wfy/code/mmdetection-master/results_val.pkl.json")
segmentation_dir="/home/wfy/code/jinnan_2/segmentation"

mkdir(segmentation_dir)
image_name = {}
npz_dic={}
for i in range(len(val_dict["images"])):
    image_name[val_dict["images"][i]["id"]] = val_dict["images"][i]["file_name"]
    height=val_dict["images"][i]["height"]
    width=val_dict["images"][i]["width"]
    npz_dic[val_dict["images"][i]["file_name"].split('.')[0]]=[np.zeros((height,width),dtype=bool)]*5
for i in range(len(load_dict)):
    image_id=(image_name[load_dict[i]["image_id"]]).split('.')[0]
    score=load_dict[i]['score']
    if score>=0.3:
        fortran_mask=decode(load_dict[i]["segmentation"])
        category_id=load_dict[i]['category_id']
        npz_dic[image_id][category_id-1]=npz_dic[image_id][category_id-1]+fortran_mask
for index_img,npz_list in npz_dic.items():
    file_name = str(index_img) + '.jpg'
    preds = []
    for i in range(len(npz_list)):
        npz_list[i]=npz_list[i].astype(np.uint8)
        npz_list[i][npz_list[i]>=1]=1
        preds.append(npz_list[i])
    preds_np = np.array(preds)  # fake prediction
    show_segmentation(file_name, preds_np, segmentation_dir)
