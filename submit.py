from pycocotools.mask import *
import json
import numpy as np
import os




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

def show_segmentation(image_name,preds,segmentation_dir):
    import cv2
    height,width=(preds.shape[1], preds.shape[2]) # (height,width)
    segmentation=np.zeros((height,width,3))
    colormap=[(127, 20, 22), (9, 128, 64), (127, 128, 51), (40, 41, 115), (125, 39, 125)]
    for cls_id in range(len(preds)):
        mask=preds[cls_id][:,:]
        print(mask.shape)
        segmentation[:,:,0]= segmentation[:,:,0]*(1-mask)+colormap[cls_id][0] * mask
        segmentation[:,:,1] = segmentation[:,:,1]*(1-mask)+colormap[cls_id][1] * mask
        segmentation[:,:,2] = segmentation[:,:,2]*(1-mask)+colormap[cls_id][2] * mask
    print(os.path.join(segmentation_dir,image_name))
    cv2.imwrite(os.path.join(segmentation_dir,image_name),segmentation)





if __name__=="__main__":
    '''
    Example code for making submit
    '''

    prediction_dir="/home/wfy/code/jinnan_2/prediction"
    segmentation_dir = "/home/wfy/code/jinnan_2/segmentation"
    json_p="./submit_10.json"

    submits_dict=dict()
    for img_id in range(1,10):
        image_name="%d.jpg"%(img_id)

        preds=[]
        for cls_id in range(1,6): # 5 classes in this competition
            cls_pred_name="%d_%d.npy"%(img_id,cls_id)
            pred_p = os.path.join(prediction_dir,cls_pred_name)
            pred=np.load(pred_p)
            preds.append(pred)

        preds_np=np.array(preds) #fake prediction
        show_segmentation(image_name, preds_np, segmentation_dir)
        submit=make_submit(image_name,preds_np)
        submits_dict[image_name]=submit

    #
    # dump_2_json(submits_dict,json_p)

