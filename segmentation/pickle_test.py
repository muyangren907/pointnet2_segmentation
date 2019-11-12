import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

if __name__ == '__main__':
    data_filename = os.path.join(DATA_DIR, 'scannet_train.pickle')

    with open(data_filename, 'rb') as fp:
        scene_points_list = pickle.load(fp, encoding='latin1')
        semantic_labels_list = pickle.load(fp, encoding='latin1')
    print(len(scene_points_list), len(semantic_labels_list))
    print(scene_points_list[0].shape, semantic_labels_list[0].shape)
    print(scene_points_list[0])
    print(semantic_labels_list[0])
