# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import json
import random
from pycocotools.mask import *
def dump_2_json(submits,save_p):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(save_p, 'w', encoding='utf-8');
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()
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
def read_file(filespath):
    files=[]
    filelist = os.listdir(filespath)
    for filename in filelist:
        filepath = os.path.join(filespath, filename)
        files.append(filepath)
    return files
def is_while_point(img,point):
    #point=[x0,y0,x1,y1]
    if list(img[point[0]][point[1]])==[255,255,255]:
        return True
    if list(img[point[0]][point[3]])==[255,255,255]:
        return True
    if list(img[point[2]][point[1]])==[255,255,255]:
        return True
    if list(img[point[2]][point[3]])==[255,255,255]:
        return True
    return False

def put_seg(bg_img,seg_img,seg_mask):
    seg_mask[seg_mask>0]=1
    height,width,__=bg_img.shape
    seg_h,seg_w,__=seg_img.shape
    x0=random.randint(height//6,height-height//6)
    y0=random.randint(width//6,width-width//6)
    x1=x0+seg_h
    y1=y0+seg_w
    while(x1>=height or y1>=width or is_while_point(bg_img,[x0,y0,x1,y1])):
        x0 = random.randint(height // 6, height - height // 6)
        y0 = random.randint(width // 6, width - width // 6)
        x1 = x0 + seg_h
        y1 = y0 + seg_w
    bg_img[x0:x1,y0:y1,:]=bg_img[x0:x1,y0:y1,:]*(1-seg_mask)+seg_img*seg_mask
    area =list(seg_mask[:,:,0].flatten()).count(1)
    blank_img=np.zeros_like(bg_img)
    blank_img[x0:x1,y0:y1]=seg_mask
    segmentation=encode(np.asfortranarray(blank_img[0]))
    bbox=[y0,x0,seg_w,seg_h]

    return segmentation,area,bbox

if __name__=="__main__":
    img_nums = 10
    seg_max_threshold=[2,2,2,2,2]
    classes = ['TieJ', 'HeiJ', 'Knief', 'Battery', 'Scissor']
    category_segmentation = "/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/category_segmentation"
    category_mask = "/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/category_mask"
    img_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/normal/'
    new_img_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/falseimg/'
    mkdir(new_img_dir)
    blank_imgs = read_file(img_dir)
    all_seg_jpg=[]
    segmentation_id=1
    json_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/train_restriction.json'
    new_json_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/falseimg.json'
    with open(json_dir,'r') as load_f:
        load_dict=json.load(load_f)
    train_dict={'info':load_dict['info'],'licenses':load_dict['licenses'],'categories':load_dict['categories'],'images':[],'annotations':[]}
    image_sample={"coco_url":"","data_captured":"","file_name":None,"flickr_url":"","id":None,"height":None,"width":None,"license":1}
    annotation_sample={"id":None,"image_id":None,"category_id":None,"iscrowd":0,"segmentation":[],"area":0.0,"bbox":None,"minAreaRect":[]}
    for cls in classes:
        category_path=os.path.join(category_segmentation,cls)
        all_seg_jpg.append(os.listdir(category_path))
    for img_num in range(img_nums):

        img_random_seed=random.randint(0,len(blank_imgs))
        print(blank_imgs[img_random_seed])
        img=cv2.imread(blank_imgs[img_random_seed])

        image_dict = image_sample.copy()
        height,width,__=img.shape
        image_dict['file_name']="%s.jpg"%(img_num)
        image_dict['id']=img_num
        image_dict['height']=height
        image_dict['width']=width
        train_dict['images'].append(image_dict)

        for cls_index in range(len(classes)):
            for i in range(random.randint(0,seg_max_threshold[cls_index])):
                seg_random_seed=random.randint(0,len(all_seg_jpg[cls_index]))
                seg_img_path=os.path.join(category_segmentation,classes[cls_index],all_seg_jpg[cls_index][seg_random_seed])
                seg_img=cv2.imread(seg_img_path)
                mask_img_path=os.path.join(category_mask,classes[cls_index],all_seg_jpg[cls_index][seg_random_seed])
                seg_mask = cv2.imread(mask_img_path)
                segmentation,area,bbox=put_seg(img,seg_img,seg_mask)

                annotation_dict=annotation_sample.copy()
                annotation_sample['id']=segmentation_id
                annotation_sample['image_id']=img_num
                annotation_dict['category_id']=cls_index+1
                annotation_dict['segmentation']=segmentation
                annotation_dict['area']=area
                annotation_dict['bbox']=bbox
                segmentation_id=segmentation_id+1
                train_dict['annotations'].append(annotation_dict)
        cv2.imwrite('%s%s.jpg'%(new_img_dir,img_num),img)
    dump_2_json(train_dict,new_json_dir)


