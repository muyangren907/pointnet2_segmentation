import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
# import cupy as np
# import minpy as np
# pclpy only support on windows

from struct import pack, unpack
import colorsys
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='kitti', help='dataset [default: kitti]')
parser.add_argument('--split', type=str, default='train', help='split [default: train]')
parser.add_argument('--sid', type=int, default=-1, help='index of scene [default: -1]')
parser.add_argument('--sid_l', type=int, default=-1, help='low index of scene [default: -1]')
parser.add_argument('--sid_h', type=int, default=-1, help='high index of scene [default: -1]')
parser.add_argument('--num_c', type=int, default=-1, help='number of classes [default: -1]')
parser.add_argument('--num_s', type=int, default=1000, help='number of scenes per file [default: 1000]')
parser.add_argument('--sleep_t', type=int, default=0, help='sleep time [default: 0]')
FLAGS = parser.parse_args()

DATASET = FLAGS.data
SPLIT = FLAGS.split
SID = FLAGS.sid
SID_L = FLAGS.sid_l
SID_H = FLAGS.sid_h
NUM_CLASSES = FLAGS.num_c
NUM_SCENCE = FLAGS.num_s
SLEEP_T = FLAGS.sleep_t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

if NUM_CLASSES == -1:
    if DATASET == 'kitti':
        label_str = ['Pedestrian', 'Person_sitting', 'Car', 'Van', 'Truck', 'Tram', 'Cyclist', 'Misc', 'DontCare']
        NUM_CLASSES = 9
    elif DATASET == 'scannet':
        label_str = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                     'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                     'otherfurniture', 'misc']
        NUM_CLASSES = 21

DET = 0


def read_data():
    data_filename = os.path.join(DATA_DIR, DATASET, '%s_%s.pickle' % (DATASET, SPLIT))
    print('Load %s' % data_filename)

    try:
        with open(data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)
    except Exception as e:
        # print(e)
        with open(data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp, encoding='latin1')
            semantic_labels_list = pickle.load(fp, encoding='latin1')
    return scene_points_list, semantic_labels_list


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


# 获取高区分度的num个颜色
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def generate_rgb(rgb_num):
    rgb_list = []
    rgb_colors = ncolors(rgb_num)
    # 确保随机性
    # rl, gl, bl = random.sample(range(0, 256), rgb_num), random.sample(range(0, 256), rgb_num), random.sample(
    #     range(0, 256), rgb_num)
    for rgbcolor in rgb_colors:
        # r, g, b = random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)
        # rgb = (rgbcolor[0] << 16 | rgbcolor[1] << 8 | rgbcolor[2])
        # b = pack('i', rgb)
        # frgb = unpack('f', b)[0]
        frgb = rgb2frgb(rgbcolor[0], rgbcolor[1], rgbcolor[2])
        rgb_list.append(frgb)
    return rgb_list


def frgb2rbg(frgb):
    b = pack('f', frgb)
    rgb = unpack('i', b)[0]
    r, g, b = (rgb >> 16) & 0xff, (rgb >> 8) & 0xff, rgb & 0xff
    return r, g, b


def rgb2frgb(r, g, b):
    rgb = (r << 16 | g << 8 | b)
    b = pack('i', rgb)
    frgb = unpack('f', b)[0]
    return frgb


def label_pic(DATASET):
    rgb_list_path = os.path.join(DATA_DIR, 'PCD', DATASET)
    if not os.path.exists(rgb_list_path):
        os.makedirs(rgb_list_path)
    rgb_list_file = os.path.join(rgb_list_path, '%s_rgb.pickle' % DATASET)

    with open(rgb_list_file, 'rb') as fp:
        rgb_list = pickle.load(fp)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # 创建子图

    rgb_classes = len(rgb_list)
    for i in range(rgb_classes):
        frgb = rgb_list[i]
        r, g, b = frgb2rbg(frgb)
        rect = plt.Rectangle((1, i), 0.5, 0.5, color=(r / 255, g / 255, b / 255))
        ax.add_patch(rect)
        plt.text(2, i, str(i))
        plt.text(5, i, label_str[i])

    plt.axis("equal")
    plt.axis('off')
    plt_save_file = os.path.join(rgb_list_path, '%s.png' % DATASET)
    plt.savefig(plt_save_file)
    print('save', plt_save_file, 'succeed')
    plt.clf()


# 输入：一个场景点云三维坐标 和 标签
# 输出：pcd彩色文件
def data2pcd(scene_points, semantic_labels):
    label_list = np.unique(semantic_labels)
    rgb_list_path = os.path.join(DATA_DIR, 'PCD', DATASET)
    if not os.path.exists(rgb_list_path):
        os.makedirs(rgb_list_path)
    rgb_list_file = os.path.join(rgb_list_path, '%s_rgb.pickle' % DATASET)
    if os.path.exists(rgb_list_file):
        with open(rgb_list_file, 'rb') as fp:
            rgb_list = pickle.load(fp)
    else:
        print('Generate RGB file')
        rgb_list = generate_rgb(NUM_CLASSES)
        print('Save RGB file at %s' % rgb_list_file)
        with open(rgb_list_file, 'wb') as fp:
            pickle.dump(rgb_list, fp)

    label_pic_file = os.path.join(rgb_list_path, '%s.png' % DATASET)
    if not os.path.exists(label_pic_file):
        label_pic(DATASET)

    semantic_labels_len = semantic_labels.shape[0]
    zeros = np.zeros(semantic_labels_len)
    semantic_labels_tmp = np.c_[semantic_labels, zeros].astype(np.float32)
    # for i in range(label_classes):
    for i in label_list:
        chose = semantic_labels_tmp[:, 0] == i
        semantic_labels_tmp[chose, 1] = rgb_list[i]
    rgb_info = semantic_labels_tmp[:, 1]
    scene_points_o = np.c_[scene_points, rgb_info]
    # print(scene_points_o)
    conststr = 'VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %s\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %s\nDATA ascii\n' % (
        semantic_labels_len, semantic_labels_len)
    save_path = os.path.join(DATA_DIR, 'PCD', DATASET, SPLIT)
    # save_file = os.path.join(DATA_DIR, 'PCD', DATASET, SPLIT, '%06d.pcd' % SID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, '%06d.pcd' % (SID + DET))
    with open(save_file, 'w') as sf:
        sf.write(conststr)
        np.savetxt(sf, scene_points_o)


def pcdview(file_path):
    import pclpy
    from pclpy import pcl
    # 读取pcd文件
    # 实例化一个指定类型的点云对象，并将文件读到对象里
    # obj = pclpy.pcl.PointCloud.PointXYZRGBA()
    obj = pclpy.pcl.PointCloud.PointXYZ()
    pcl.io.loadPCDFile(file_path, obj)

    # 显示点云
    viewer = pcl.visualization.PCLVisualizer('PCD viewer')
    # 设置初始视角，可不写 viewer.setCameraPosition(0,0,-3.0,0,-1,0)
    # 设置显示坐标轴，可不写 viewer.addCoordinateSystem(0.5)
    viewer.addPointCloud(obj)
    while (not viewer.wasStopped()):
        viewer.spinOnce(100)


if __name__ == '__main__':
    scene_points_list, semantic_labels_list = read_data()
    if '_' in SPLIT:
        DET = int(SPLIT.split('_')[1]) * NUM_SCENCE
        SPLIT = SPLIT.split('_')[0]
    list_len = len(scene_points_list)
    l, h = 0, list_len
    del_c = 0
    if SID == -1:
        if SID_L != -1:
            l = SID_L
        if SID_H != -1:
            h = SID_H
        for SID in range(l, h):
            print('[', SID + 1, '/', list_len, ']', end='\r')
            data2pcd(scene_points_list[SID], semantic_labels_list[SID])
            del_c += 1
            if del_c == 10:
                del_c = 0
                print('sleep', SLEEP_T, 's')
                time.sleep(SLEEP_T)

    else:
        data2pcd(scene_points_list[SID], semantic_labels_list[SID])
