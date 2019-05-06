import json

def read_json(filename):
    with open(filename,'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

def write_json(filename,file_dict):
    with open(filename,"w") as f:
        json.dump(file_dict,f)

if __name__ == "main":
    val_dict = read_json("/home/wfy/code/jinnan4/test.json")
    print(val_dict)
    load_dict = read_json("/home/wfy/code/mmdetection-master/output.pkl.json")
    image_name = {}
    for i in range(len(val_dict["images"])):
        image_name[val_dict["images"][i]["id"]] = val_dict["images"][i]["file_name"]
    # 按照模板生成一个空的容器result_dict
    
    results_list  = []
    for i in range(len(image_name)):
        single_file_dict = {"filename":image_name[i],"rects":[]}
        results_list.append(single_file_dict)
    
    
    # 然后读取mmdetection的结果并放入到容器中
    for i in range(len(load_dict)):
        xmin = load_dict[i]['bbox'][0]
        ymin = load_dict[i]['bbox'][1]
        xmax = xmin + load_dict[i]['bbox'][2]
        ymax = ymin + load_dict[i]['bbox'][3]
        label = load_dict[i]['category_id']
        confidence = load_dict[i]['score']
        
        
        temp_dict = {"xmin":xmin , "xmax":xmax , "ymin": ymin, "ymax": ymax, "label": label, "confidence": confidence}
        if results_list[load_dict[i]['image_id']]['filename'] == image_name[load_dict[i]['image_id']]:
            results_list[load_dict[i]['image_id']]['rects'].append(temp_dict)
    result_dict = {'results':results_list}
    
    write_json("/home/wfy/code/mmdetection-master/320.json",result_dict)

