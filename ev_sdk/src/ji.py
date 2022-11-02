import cv2
import numpy as np
import json
import torch
from model.centernet3d import CenterNet3D
from utils import deal_with_image,get_calib,extract_dets_from_outputs,decode_detections,Calibration

def init():       # 模型初始化
    model_path = "/project/train/models/checkpoint_epoch_9cls_half_10.pth"
    model = CenterNet3D(backbone='dla34', neck='DLAUp', downsample=4, num_class=9)
    model.load_state_dict(torch.load(model_path)['model_state'])
    model=model.to("cuda")
    return model

def process_image(net, input_image, args=None):
    with torch.no_grad():
        info = {'img_id': [0], 'img_size': [[1920//2, 1080//2]], 'bbox_downsample_ratio': [[4.0000 * 2, 3.9706 * 2]]}
        class_list = ['car','van' ,'truck','bus','pedestrian','cyclist','motorcyclist', 'barrow' ,'tricyclist']
        cls_mean_size = np.zeros((len(class_list), 3))
        args = json.loads(args)
        calib = args.get('calib')
        depth = args.get('depth')
        # print("depth:",depth)
        # calib = "2184.084907 0.000000 990.488434 0 0.000000 2330.131664 541.755886 0 0.000000 0.000000 1.000000 0"
        calibs = [Calibration(get_calib(calib))]

        depth_img = cv2.imread(depth,cv2.IMREAD_GRAYSCALE)
        input_image = input_image[:, :, ::-1]
        input_image = deal_with_image(input_image,depth_img).to("cuda")
        dets = net(input_image)
        dets = extract_dets_from_outputs(outputs=dets,depth_img_o=depth_img, K=50)
        dets = dets.detach().cpu().numpy()

        # get corresponding calibs & transform tensor to numpy
        dets = decode_detections(dets=dets,
                                info=info,
                                calibs=calibs,
                                cls_mean_size=cls_mean_size,
                                threshold=0.35)
        fine2coarse = {}
        fine2coarse['van'] = 'car'
        fine2coarse['car'] = 'car'
        fine2coarse['bus'] = 'big_vehicle'
        fine2coarse['truck'] = 'big_vehicle'
        fine2coarse['cyclist'] = 'cyclist'
        fine2coarse['motorcyclist'] = 'cyclist'
        fine2coarse['tricyclist'] = 'cyclist'
        fine2coarse['pedestrian'] = 'pedestrian'
        fine2coarse['barrow'] = 'pedestrian'
        temp1 = dets[0]
        results = []
        for i in range(len(temp1)):
            temp2 = [str(x) for x in temp1[i]]
            temp2[0] = class_list[int(temp2[0])]
            temp2[0] = fine2coarse[temp2[0]]
            results.append(temp2)
        '''
            假如 dets数据为
            dets = [['pedestrian', '0', '0', '1.9921350550721388', '783.46228', '256.193909', '807.102356', '305.991242', '1.501095', '0.443502', '0.599597', '-6.30840572763', '-7.24883889889', '74.6925154711', '1.90787668151'], 
                    ['car', '0', '0', '-1.4957292038451553', '1513.208984', '221.756866', '1582.604248', '285.019531', '1.138008', '1.533378', '4.259202', '15.9548793918', '-6.77630765403', '61.575159005', '-1.24219284167'], 
                    ['car', '0', '1', '1.819301750914212', '1047.92334', '258.272095', '1138.018677', '326.640473', '1.201635', '1.402634', '4.360323', '3.16898439105', '-5.66824359768', '61.0786524493', '1.87113893629']]
            数据含义与数据集中标注含义一致
        * 如需，可以在args参数中获取calib等参数文件，返回内容为文件路径：
                args = json.loads(args)
                calib = args.get('calib')
                denorm = args.get('denorm')
                extrinsics = args.get('extrinsics')
                depth = args.get('depth')
        '''
        result = {"model_data": {"objects": results}}
        return json.dumps(result)


if __name__ == '__main__':
    predictor = init()
    original_image = cv2.imread('xxx.jpg')   # 读取图片
    result = process_image(predictor, original_image)
    print(result)
