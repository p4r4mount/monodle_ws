import torch
import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils import Calibration,Object3d,affine_transform,get_affine_transform,gaussian_radius,draw_umich_gaussian,angle2class
from draw_bbox import draw_bbox

class Rope_Dataset(Dataset):
    def __init__(self, mode, downsample,depth_threshold):
        data_pth = '../../data'
        # data_pth = 'home/data'
        data_ls = os.listdir(data_pth)
        test_pth = os.path.join(data_pth,data_ls[0])
        self.depth_threshold = depth_threshold

        self.mode = mode
        assert self.mode in['train','val']
        self.writelist = ['car','van' ,'truck','bus','pedestrian','cyclist','motorcyclist', 'barrow' ,'tricyclist']
        self.cls2id = {'car':0,'van':1 ,'truck':2,'bus':3,'pedestrian':4,'cyclist':5,'motorcyclist':6, 'barrow':7 ,'tricyclist':8}
        self.calib_pth = os.path.join(data_pth,data_ls[0],'calib')
        self.image_pth = os.path.join(data_pth,data_ls[0],'image')
        self.label_pth = os.path.join(data_pth,data_ls[0],'label')
        # self.depth_pth = os.path.join(data_pth,data_ls[1],'depth')
        # self.depth_ls = sorted(os.listdir(self.depth_pth))
        self.calib_ls = sorted(os.listdir(self.calib_pth))
        self.image_ls = sorted(os.listdir(self.image_pth))
        self.label_ls = sorted(os.listdir(self.label_pth))
        self.use_3d_center = True
        self.max_objs = 50
        self.num_classes = len(self.writelist)
        self.downsample = downsample
        self.data_augmentation = True
        self.random_flip = 0.5
        self.random_crop = 0.2
        self.scale = 0.4
        self.shift = 0.1

        # depth_img = cv2.imread(os.path.join(self.depth_pth,self.depth_ls[0]))
        image_img = cv2.imread(os.path.join(self.image_pth,self.image_ls[0]))
        self.resolution = np.array([image_img.shape[1],image_img.shape[0]])
        self.resolution = np.array([1920,1088])
        # print(self.resolution)
        # print(self.calib_ls[0], self.image_ls[0])

        # data_name = np.loadtxt(test_pth+'/sample.txt', dtype=str)
        # print(len(data_name))
        # for i in range(len(data_name)):
        #     print(calib_ls[i].split('.')[0]==data_name[i])

        # print(test_pth,depth_ls[0])
        # calib = np.loadtxt(calib_pth,dtype=str)

    def get_calib(self,i):
        calib_file = os.path.join(self.calib_pth,self.calib_ls[i])
        return(Calibration(calib_file))# 返回calib类

    def get_image(self,i):
        img_file = os.path.join(self.image_pth,self.image_ls[i])
        return Image.open(img_file)

    def get_label(self,i):
        label_file = os.path.join(self.label_pth,self.label_ls[i])
        with open(label_file, 'r') as f:
            lines = f.readlines()
        objects = [Object3d(line) for line in lines]
        return(objects)# 返回目标列表

    def __len__(self):
        return len(self.calib_ls)

    def __getitem__(self,i):
        # image loading 获取图像
        img = self.get_image(i)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample
        # print(features_size)

        # 图像增强 data augmentation for image
        center = img_size / 2
        aug_scale, crop_size = 1.0, img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            if np.random.random() < self.random_crop:
                random_crop_flag = True
                aug_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                crop_size = img_size * aug_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            
        # 2D图像的放射变换
        # print(aug_scale,crop_size,center)
        # fuck = np.array([1920,1088])
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        
        img = np.array(img).astype(np.float32) / 255.0
        # print(img.shape)
        # 归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        cls_mean_size = np.zeros((9,3), dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        # print(img.shape)

        info = {'img_id': i,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}
        
        # 获取标签
        objects = self.get_label(i)
        # print(objects[0].to_kitti_format().split(' ')[0])
        calib = self.get_calib(i)

        # 标签的增强
        if random_flip_flag:
            for object in objects:
                [x1, _, x2, _] = object.box2d
                # print(x1,x2)
                object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi
        
        # 标签encoding
        heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
        mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        # print(object_num)

        for i in range(object_num):
            # print(objects[i].box2d)
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                # print(1)
                # print(objects[i].cls_type)
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                # print(2)
                # print(objects[i].cls_type)
                # print(objects[i].level_str)
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = self.depth_threshold
            if objects[i].pos[-1] > threshold:
                # print(3)
                # print(objects[i].cls_type)
                # print(objects[i].pos[-1])
                continue
            
            # print(objects[i].box2d)
            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d[:] /= self.downsample

            # process 3d bbox & get 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)
            center_3d /= self.downsample

            # generate the center of gaussian heatmap [optional: 3d center or 2d center]
            center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
            if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
            if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

            # generate the radius of gaussian heatmap
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                continue

            cls_id = self.cls2id[objects[i].cls_type]
            draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

            # encoding 2d/3d offset & 2d size
            indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
            offset_2d[i] = center_2d - center_heatmap
            size_2d[i] = 1. * w, 1. * h

            # encoding depth
            depth[i] = objects[i].pos[-1] * aug_scale

            # encoding heading angle
            heading_angle = objects[i].alpha
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding 3d offset & size_3d
            offset_3d[i] = center_3d - center_heatmap
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            mask_2d[i] = 1
            mask_3d[i] = 0 if random_crop_flag else 1
        
        # 整合返回数据
        inputs = img
        targets = {'depth': depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'mask_3d': mask_3d}

        return inputs, targets, info





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = Rope_Dataset('train',downsample=8,depth_threshold=120)
    dataloader = DataLoader(dataset=dataset,batch_size=1)

    for i,(inputs,target,info) in enumerate(dataloader):
        print(target['size_3d'].max())
        break

            
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
        break