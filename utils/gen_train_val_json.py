# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:20:54 2019

@author: dongl
"""
# 此脚本用于生成训练和验证的json文件
import numpy as np
import json 
import random

# # fixbug
# jsonDir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/'
# with open(jsonDir+"train_normalization.json",'r') as load_f:
#     load_dict = json.load(load_f)
# print(load_dict.keys())
# annotations = load_dict["annotations"]
# for i in range(len(annotations)):
#     annotations[i]["area"] = annotations[i]["bbox"][2]*annotations[i]["bbox"][3]
# train_dict = {'info':load_dict["info"],'licenses':load_dict["licenses"],'categories':load_dict["categories"],
#               'images':load_dict["images"],'annotations':annotations}

# with open(jsonDir+"train_restriction.json","w") as f:
#     json.dump(train_dict,f)
# # generate train and val
# with open(jsonDir+"train_restriction.json",'r') as load_f:
#     load_dict = json.load(load_f)
# print(load_dict.keys())


# indexList = range(len(load_dict["images"]))
# train_list = random.sample(indexList, 2800)
# val_list = list(set(indexList)-set(train_list))
#
#
# train_images_list = []
# val_images_list = []
# train_annotations_list = []
# val_annotations_list = []
# for i in indexList:
#     if load_dict["images"][i]["id"] in train_list:
#         train_images_list.append(load_dict["images"][i])
#     if load_dict["images"][i]["id"] in val_list:
#         val_images_list.append(load_dict["images"][i])
# for i in range(len(load_dict["annotations"])):
#     if load_dict["annotations"][i]["image_id"] in train_list:
#         train_annotations_list.append(load_dict["annotations"][i])
#     if load_dict["annotations"][i]["image_id"] in val_list:
#         val_annotations_list.append(load_dict["annotations"][i])
#
# train_dict = {'info':load_dict["info"],'licenses':load_dict["licenses"],'categories':load_dict["categories"],
#               'images':train_images_list,'annotations':train_annotations_list}
# val_dict = {'info':load_dict["info"],'licenses':load_dict["licenses"],'categories':load_dict["categories"],
#               'images':val_images_list,'annotations':val_annotations_list}
# with open(jsonDir+"normal_train.json","w") as f:
#     json.dump(train_dict,f)
# with open(jsonDir+"normal_val.json","w") as f:
#     json.dump(val_dict,f)

# with open('/home/wfy/code/Detectron.pytorch/data/jinnan/train.json') as load_f:
#     load_dict = json.load(load_f)
# annotations = load_dict["annotations"]
# for i in range(len(annotations)):
#     annotations[i]["segmentation"] = annotations[i]["minAreaRect"]
# train_dict = {'info':load_dict["info"],'licenses':load_dict["licenses"],'categories':load_dict["categories"],
#               'images':load_dict["images"],'annotations':annotations}
# with open("/home/wfy/code/Detectron.pytorch/data/jinnan/train_mask.json","w") as f:
#     json.dump(train_dict,f)

with open("/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/train_restriction.json",'r') as load_f:
    load_dict = json.load(load_f)
print(load_dict.keys())
annotations = load_dict["annotations"]
images=load_dict['images']
set=set()
for i in range(len(annotations)):
    id=annotations[i]['image_id']
    set.add(id)
list=[]
for i in range(2025):
    if i not in set:
        list.append(i)
for i in range(len(images)):
    id=images[i]['id']
    if id in list:
        print(images[i]['file_name'])

# with open(jsonDir+"val.json",'r') as load_f:
#     load_dict = json.load(load_f)
# annotations = load_dict["annotations"]
# for i in range(len(annotations)):
#     list=[]
#     for m in annotations[i]['minAreaRect']:
#         for n in m:
#             list.append(n)
#     annotations[i]["segmentation"] = [list]
# train_dict = {'info':load_dict["info"],'licenses':load_dict["licenses"],'categories':load_dict["categories"],
#                'images':load_dict["images"],'annotations':annotations}
# with open(jsonDir+"val_mask.json","w") as f:
#     json.dump(train_dict,f)