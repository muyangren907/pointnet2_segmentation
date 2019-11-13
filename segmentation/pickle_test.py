import pickle
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='scannet', help='dataset')
parser.add_argument('--split', type=str, default='train', help='split')
FLAGS = parser.parse_args()

DATASET = FLAGS.data
SPLIT = FLAGS.split

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

    # m, n = 999, -1
    #
    # for label in semantic_labels_list:
    #     a, b = np.max(label), np.min(label)
    #     print(m, a, n, b)
    #     m, n = max(a, m), min(b, n)
    for label in semantic_labels_list:
        tmp, _ = np.histogram(label, range(22))
        print(tmp)
        print(_)
