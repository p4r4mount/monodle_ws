import os
import cv2
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
from model import dla
from model.dlaup import DLAUp
from utils import Calibration,decode_detections,extract_dets_from_outputs,get_calib,deal_with_image
from model.centernet3d import CenterNet3D
from draw_bbox import draw_bbox

def my_test(image_path,model_path,calib,score_threshold):
    class_list = ['car', 'van', 'truck', 'bus', 'pedestrian', 'cyclist', 'motorcyclist', 'barrow', 'tricyclist']
    image = Image.open(image_path)
    image = deal_with_image(image)
    calibs = [Calibration(get_calib(calib))]
    info = {'img_id': [0], 'img_size': [[1920, 1080]], 'bbox_downsample_ratio': [[4.0000*2, 3.9706*2]]}
    cls_mean_size = np.zeros((9, 3))
    net = CenterNet3D(backbone='dla34', downsample=8, num_class=9)
    net.load_state_dict(torch.load(model_path)['model_state'])

    outputs = net(image)
    dets = extract_dets_from_outputs(outputs=outputs, K=50)
    dets = dets.detach().cpu().numpy()

    # get corresponding calibs & transform tensor to numpy
    dets = decode_detections(dets=dets,
                             info=info,
                             calibs=calibs,
                             cls_mean_size=cls_mean_size,
                             threshold=score_threshold)
    temp1 = dets[0]
    results = []
    for i in range(len(temp1)):
        temp2 = [str(x) for x in temp1[i]]
        temp2[0] = class_list[int(temp2[0])]
        results.append(temp2)
        # print(temp2[0])
    # print(results)
    label_list = [line[0] for line in results]
    bbox_list = [line[4:8] for line in results]
    bbox_list = [list(map(float, bbox)) for bbox in bbox_list]
    # print(label_list,bbox_list)
    
    draw_bbox(image_path, label_list, bbox_list,save_pth)


if __name__ == '__main__':
    class_list = ['car','van' ,'truck','bus','pedestrian','cyclist','motorcyclist', 'barrow' ,'tricyclist']
    # calib = "2757.839717 0.000000 992.437197 0 0.000000 2899.073935 576.566446 0 0.000000 0.000000 1.000000 0"
    calib = "2184.084907 0.000000 990.488434 0 0.000000 2330.131664 541.755886 0 0.000000 0.000000 1.000000 0"
    image_path = '/home/song/Proj/Rope3D/monodle_ws/data/1/image/1784_fa2sd4adatasetWest152_420_1621243598_1621243749_6_obstacle.jpg'
    model_path = '/home/song/Proj/Rope3D/monodle_models/wusongzuiaidemoxing.pth'
    save_pth = '../result-graphs/a.jpg'
    my_test(image_path=image_path,model_path=model_path,calib=calib,score_threshold=0.2)