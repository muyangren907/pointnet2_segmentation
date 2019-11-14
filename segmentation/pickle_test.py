import pickle
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='scannet', help='dataset')
parser.add_argument('--split', type=str, default='train', help='split')
parser.add_argument('--num_c', type=int, default=21, help='num_classes')
FLAGS = parser.parse_args()

DATASET = FLAGS.data
SPLIT = FLAGS.split
NUM_CLASSES = FLAGS.num_c

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

if __name__ == '__main__':
    data_filename = os.path.join(DATA_DIR, DATASET, '%s_%s.pickle' % (DATASET, SPLIT))

    try:
        with open(data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)
    except Exception as e:
        # print(e)
        with open(data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp, encoding='latin1')
            semantic_labels_list = pickle.load(fp, encoding='latin1')
    print(len(scene_points_list), len(semantic_labels_list))
    print(scene_points_list[0].shape, semantic_labels_list[0].shape)
    print(scene_points_list[0])
    print(semantic_labels_list[0])
    input('continue')
    if DATASET == 'scannet':
        NUM_CLASSES = 21
    elif DATASET == 'kitti':
        NUM_CLASSES = 9
    # m, n = 999, -1
    #
    # for label in semantic_labels_list:
    #     a, b = np.max(label), np.min(label)
    #     print(m, a, n, b)
    #     m, n = max(a, m), min(b, n)
    a = 'r'
    label_count = np.zeros(NUM_CLASSES)
    for label in semantic_labels_list:
        # 统计0-21各有多少个
        """ 
        tmp
        [32198 12867 25137     0  1554     0     0     0     0     0     0     0     0     0  1917     0     0     0     0     0     0]
        _
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
        """
        tmp, _ = np.histogram(label, range(NUM_CLASSES + 1))
        label_count += tmp
        # print(tmp)
        # # print(_)
        # a = input('continue')
        # if a == 'q':
        #     break
    print(label_count)
    print(np.sum(label_count))
    print()
    a = np.array([1, 1, 2])
    b = np.array([1, 2, 3])
    c = a + b
    print(a)
    print(b)
    print(c)
    print(c - 2)
