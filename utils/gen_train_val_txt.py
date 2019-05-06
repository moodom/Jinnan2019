from PIL import Image
import os
def readFilename(path):
    allfile=[]
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        allfile.append(filepath)
    return allfile

if __name__=='__main__':
    res_image_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/binary/train/restricted_train'
    nor_image_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/binary/train/normal_train'
    res_image_list=readFilename(res_image_dir)
    nor_image_list=readFilename(nor_image_dir)

    txt_dir='/home/wfy/code/jinnan_2/jinnan2_round2_train_20190401/binary/train/train.txt'
    with open(txt_dir,'w') as f:
        for i in res_image_list:
            f.write(str(i)+' 1\n')
        for i in nor_image_list:
            f.write(str(i)+' 0\n')