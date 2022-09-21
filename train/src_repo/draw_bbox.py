import cv2
import numpy as np



def draw_bbox(img_pth,label_list,bbox_list,save_pth):
    # 边框格式　bbox = [xl, yl, xr, yr]
    # label = 'man'
    image = cv2.imread(img_pth)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(label_list)):

        bbox = list(map(int,bbox_list[i]))
        label = label_list[i]

        label_size = cv2.getTextSize(label, font, 1, 2)
        # 设置label起点
        text_origin1 = np.array([bbox[0], bbox[1] - label_size[0][1]])
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,255), thickness = 2 )
        cv2.rectangle(image, tuple(text_origin1), tuple(text_origin1 + label_size[0]), color=(0, 0, 255), thickness = -1)  # thickness=-1 表示矩形框内颜色填充
        # 1为字体缩放比例，2表示自体粗细
        cv2.putText(image, label, (bbox[0], bbox[1] - 5), font, 1, (255, 255, 255), 2)

    cv2.imwrite(save_pth,image)


def get_label(label_pth):
    with open(label_pth, 'r') as f:
        lines = f.readlines()
        label_list = [line.split(' ')[0] for line in lines]
        bbox_list = [line.split(' ')[4:8] for line in lines]
        bbox_list = [list(map(float, bbox)) for bbox in bbox_list]
    return label_list,bbox_list



if __name__ == '__main__':
    img_pth = '/home/song/Proj/Rope3D/monodle_ws/data/1/image/1784_fa2sd4adatasetWest152_420_1621243598_1621243749_6_obstacle.jpg'  
    label_pth = '/home/song/Proj/Rope3D/monodle_ws/data/1/label/1784_fa2sd4adatasetWest152_420_1621243598_1621243749_6_obstacle.txt'
    save_pth = 'a.jpg'
    label_list,bbox_list = get_label(label_pth)
    draw_bbox(img_pth, label_list, bbox_list,save_pth)