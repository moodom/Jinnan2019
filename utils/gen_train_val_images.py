import os
import random
import cv2

def readFilename(path):
    allfile=[]
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        allfile.append(filepath)
    return allfile

if __name__=="__main__":
    data_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/restricted'
    save_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/binary/'
    indexList=readFilename(data_dir)
    train_list = random.sample(indexList, 1825)
    val_list = list(set(indexList)-set(train_list))
    for name in train_list:
        img=cv2.imread(name)
        train_save_path=save_dir+'restricted_train/'+name.split('/')[7]
        print(train_save_path)
        cv2.imwrite(train_save_path,img)
    for name in val_list:
        img=cv2.imread(name)
        train_save_path=save_dir+'restricted_val/'+name.split('/')[7]
        cv2.imwrite(train_save_path,img)
