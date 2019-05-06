import json
import numpy as np
from pycocotools.mask import *
import cv2
def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def write_json(filename, file_dict):
    with open(filename, "w") as f:
        json.dump(file_dict, f)

# def read_txt(file_path):
#     Data=[]
#     with open(file_path) as txtData:
#         lines=txtData.readlines()
#         for line in lines:
#             line_data=line.strip().split('.')[0]
#             Data.append(line_data)
#     return Data
#
# binary_data=read_txt('/home/wfy/code/mmdetection-master/restri2_a.txt')
# print(binary_data)

val_dict = read_json("/home/wfy/code/jinnan_2/test_round2_a.json")
load_dict = read_json("/home/wfy/code/mmdetection-master/mask_100/results.pkl.json")
binary_dict=read_json("/home/wfy/code/mmdetection-master/test_dict_a.json")
save_npz_dir="/home/wfy/code/jinnan_2/prediction/"
segmentation_dir="/home/wfy/code/jinnan_2/segmentation/"


image_name = {}
npz_dic={}
for i in range(len(val_dict["images"])):
    image_name[val_dict["images"][i]["id"]] = val_dict["images"][i]["file_name"]
    height=val_dict["images"][i]["height"]
    width=val_dict["images"][i]["width"]
    npz_dic[val_dict["images"][i]["file_name"].split('.')[0]]=[np.zeros((height,width),dtype=bool)]*5
for i in range(len(load_dict)):
    if binary_dict[image_name[load_dict[i]["image_id"]]]==1:
    # if image_name[load_dict[i]["image_id"]] in binary_data:
        image_id=(image_name[load_dict[i]["image_id"]]).split('.')[0]
        fortran_mask=decode(load_dict[i]["segmentation"])
        category_id=load_dict[i]['category_id']
        npz_dic[image_id][category_id-1]=npz_dic[image_id][category_id-1]+fortran_mask
for index_img,npz_list in npz_dic.items():
    print(index_img)
    for i in range(len(npz_list)):
        npz_list[i]=npz_list[i].astype(np.uint8)
        npz_list[i][npz_list[i]>=1]=1
        np.save('%s%s_%s'%(save_npz_dir,index_img,(i+1)),npz_list[i])
#         #cv2.imwrite('%s%s_%s.jpg'%(segmentation_dir,index_img,(i+1)),npz_list[i]*255)

