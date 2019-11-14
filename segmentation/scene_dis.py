import pickle
import os
import argparse
import numpy as np
import pclpy
from pclpy import pcl
from struct import pack, unpack
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='scannet', help='dataset')
parser.add_argument('--split', type=str, default='train', help='split')
parser.add_argument('--sid', type=int, default=0, help='index of scene')
parser.add_argument('--num_c', type=int, default=21, help='num_classes')
FLAGS = parser.parse_args()

DATASET = FLAGS.data
SPLIT = FLAGS.split
SID = FLAGS.sid
NUM_CLASSES = FLAGS.num_c

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')


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


def generate_rgb(rgb_num):
    rgb_list = []
    for i in range(rgb_num):
        r, g, b = random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)
        rgb = (r << 16 | g << 8 | b)
        b = pack('i', rgb)
        frgb = unpack('f', b)[0]
        rgb_list.append(frgb)
    return rgb_list


# 输入：一个场景点云三维坐标 和 标签
# 输出：pcd彩色文件
def data2pcd(scene_points, semantic_labels):
    # pass
    # obj = pclpy.pcl.PointCloud.PointXYZRGB()
    label_count = np.bincount(semantic_labels)
    label_classes = len(label_count)
    rgb_list = generate_rgb(label_classes)
    semantic_labels_len = semantic_labels.shape[0]
    zeros = np.zeros(semantic_labels_len)
    semantic_labels_tmp = np.c_[semantic_labels, zeros].astype(np.float32)
    for i in range(label_classes):
        chose = semantic_labels_tmp[:, 0] == i
        semantic_labels_tmp[chose, 1] = rgb_list[i]
    rgb_info = semantic_labels_tmp[:, 1]
    scene_points_o = np.c_[scene_points, rgb_info]
    print(scene_points_o)
    conststr = 'VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %s\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %s\nDATA ascii\n' % (
        semantic_labels_len, semantic_labels_len)
    save_file = os.path.join(DATA_DIR, DATASET, '%s_%06d.pcd' % (SPLIT, SID))
    with open(save_file, 'w') as sf:
        sf.write(conststr)


def pcdview(file_path):
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
    # r = 109
    # g = 114
    # b = 134
    # rgb = (r << 16 | g << 8 | b)
    # print(rgb)
    # b = pack('i', rgb)
    # print(unpack('f', b)[0])
    # a = 1
    # b = 1.000
    # print(a, b, a == b)
    scene_points_list, semantic_labels_list = read_data()
    data2pcd(scene_points_list[SID], semantic_labels_list[SID])
