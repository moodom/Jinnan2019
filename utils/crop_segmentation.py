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
if __name__=="__main__":
    classes_nums = {1:0, 2:0, 3:0, 4:0, 5:0}
    classes = ['TieJ', 'HeiJ', 'Knief', 'Battery', 'Scissor']
    json_dir="/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/train.json"
    category_segmentation = "/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/category_segmentation"
    category_mask='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/category_mask'
    img_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/restricted'
    mkdir(category_segmentation)
    mkdir(category_mask)
    for cls in classes:
        mkdir(os.path.join(category_segmentation,cls))
    for cls in classes:
        mkdir(os.path.join(category_mask,cls))
    npz_dic={}
    with open(json_dir,'r') as load_f:
        load_dict=json.load(load_f)
    for i in range(len(load_dict['annotations'])):
        category_id=load_dict['annotations'][i]['category_id']
        image_id=load_dict['annotations'][i]['image_id']
        segmentation=load_dict['annotations'][i]['segmentation']
        bbox=load_dict['annotations'][i]['bbox']
        file_name = str(image_id) + '.jpg'
        img=cv2.imread(os.path.join(img_dir,file_name))
        height,width,__=img.shape
        fortran_mask = decode(maskUtils.frPyObjects(segmentation, height, width))
        img_segmentation=img
        crop_segmentation=img_segmentation[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
        crop_mask=fortran_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
        cv2.imwrite('%s/%s/%s.jpg'%(category_segmentation,classes[category_id-1],str(classes_nums[category_id])),crop_segmentation)
        cv2.imwrite('%s/%s/%s.jpg' % (category_mask, classes[category_id - 1], str(classes_nums[category_id])),
                    crop_mask)
        classes_nums[category_id]+=1