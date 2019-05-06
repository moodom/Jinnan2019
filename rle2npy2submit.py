MMD_JSON = "/home/wfy/code/mmdetection-master/results.pkl.json"
SUBMIT_JSON = "/home/wfy/code/mmdetection-master/submit_17.json"

import json
import os
import numpy as np
from pycocotools.mask import *

def read_json(filename):
    with open(filename,'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

def make_submit(image_name,preds):
    '''
    Convert the prediction of each image to the required submit format
    :param image_name: image file name
    :param preds: 5 class prediction mask in numpy array
    :return:
    '''

    submit=dict()
    submit['image_name']= image_name
    submit['size']=(preds.shape[1],preds.shape[2])  #(height,width)
    submit['mask']=dict()

    for cls_id in range(0,5):      # 5 classes in this competition

        mask=preds[cls_id,:,:]
        cls_id_str=str(cls_id+1)   # class index from 1 to 5,convert to str
        fortran_mask = np.asfortranarray(mask)
        rle = encode(fortran_mask) #encode the mask into rle, for detail see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        submit['mask'][cls_id_str]=rle

    return submit



def dump_2_json(submits,save_p):
    '''

    :param submits: submits dict
    :param save_p: json dst save path
    :return:
    '''
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(save_p, 'w', encoding='utf-8');
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()

classify_dict = read_json("/home/wfy/code/mmdetection-master/test_dict_a.json")

test_json = read_json("/home/wfy/code/jinnan_2/test_round2_a.json")
image_id_name = {}
testimage_meta = {}
for i in range(len(test_json["images"])):
    image_id_name[test_json["images"][i]["id"]] = test_json["images"][i]["file_name"]
    testimage_meta[test_json["images"][i]["file_name"]] = [test_json["images"][i]["height"],test_json["images"][i]["width"]]
    
# a list
mmd_json = read_json(MMD_JSON)
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
    

therehold = [0.2,0.08,0.15,0.2,0.1]
#therehold = [0.3,0.3,0.3,0.3,0.3]
#therehold = [0.5,0.5,0.5,0.5,0.5]
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
# ensemble binary classify result
for key in classify_dict:
    if classify_dict[key] == 0:
        temp = np.zeros(testimage_meta[key])>0
        category_dict = {
            1:temp,
            2:temp,
            3:temp,
            4:temp,
            5:temp}
        result_dict[key] = category_dict

            
if __name__=="__main__":
    '''
    Example code for making submit
    '''
    json_p=SUBMIT_JSON

    submits_dict=dict()
    for key in result_dict:
        image_name=key

        preds=[]
        for cls_id in range(1,6): # 5 classes in this competition
            pred = result_dict[key][cls_id].astype('uint8')
#            print(pred)
#            print("--------")
            preds.append(pred)

        preds_np=np.array(preds) #fake prediction
        submit=make_submit(image_name,preds_np)
        submits_dict[image_name]=submit


    dump_2_json(submits_dict,json_p)