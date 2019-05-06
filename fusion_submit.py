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
            print()
            temp = np.zeros(testimage_meta[key])>0
            category_dict = {
                1: temp,
                2: temp,
                3: temp,
                4: temp,
                5: temp}
            result_dict[key] = category_dict
    for i in range(len(json_dict)):

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
            truth_dic[image_name][1] = result_dict[image_name][1] + npy
        elif category_id == 2:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][2] = result_dict[image_name][2] + npy
        elif category_id == 3:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][3] = result_dict[image_name][3] + npy
        elif category_id == 4:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][4] = result_dict[image_name][4] + npy
        else:
            npy = seg2npy(h, w, segmentation) > 0
            truth_dic[image_name][5] = result_dict[image_name][5] + npy
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


if __name__=="__main__":
    test_json = read_json("/home/wfy/code/jinnan_2/test_round2_b.json")
    classify_dict = read_json("/home/wfy/code/mmdetection-master-old/test_dict_b.json")
    # a list
    mmd_json_1 = read_json("/home/wfy/code/mmdetection-master-old/round_2/results_720_2.pkl.json")
    mmd_json_2 = read_json("/home/wfy/code/mmdetection-master-old/round_2/results_880_2.pkl.json")
    mmd_json_3 = read_json("/home/wfy/code/mmdetection-master-old/round_2/results_1040_2.pkl.json")
    mmd_json_4 = read_json("/home/wfy/code/mmdetection-master-old/round_2/results_1200_2.pkl.json")

    SUBMIT_JSON = "/home/wfy/code/mmdetection-master-old/around2_b_2.json"

    therehold = [0.3, 0.3, 0.3, 0.3, 0.3]
    total_therehold = [1.3, 1.3, 1.3, 1.3, 1.3]
    all_json = [mmd_json_1, mmd_json_2, mmd_json_3,mmd_json_4]
    result_dict = None
    for mmd_json in all_json:
        print(mmd_json)

        result=trans_segmentation(test_json,mmd_json,therehold,result_dict)
        result_dict=result

    result_dict = threshold_result(result_dict, total_therehold)

    image_id_name = {}
    testimage_meta = {}
    for i in range(len(test_json["images"])):
        image_id_name[test_json["images"][i]["id"]] = test_json["images"][i]["file_name"]
        testimage_meta[test_json["images"][i]["file_name"]] = [test_json["images"][i]["height"],
                                                               test_json["images"][i]["width"]]
    for key in classify_dict:
        if classify_dict[key] == 0:
            temp = np.zeros(testimage_meta[key]) > 0
            category_dict = {
                1: temp,
                2: temp,
                3: temp,
                4: temp,
                5: temp}
            result_dict[key] = category_dict


    json_p = SUBMIT_JSON

    submits_dict = dict()
    for key in result_dict:
        image_name = key

        preds = []
        for cls_id in range(1, 6):  # 5 classes in this competition
            pred = result_dict[key][cls_id].astype('uint8')
            #            print(pred)
            #            print("--------")
            preds.append(pred)

        preds_np = np.array(preds)  # fake prediction
        submit = make_submit(image_name, preds_np)
        submits_dict[image_name] = submit

    dump_2_json(submits_dict, json_p)
