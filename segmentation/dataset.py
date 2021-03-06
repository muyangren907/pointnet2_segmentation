import pickle
import os
import sys
import numpy as np
import pc_util
import scene_util


# TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train',dataname=DATASET)
# TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test')
# TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')
# NUM_CLASSES

class Dataset():
    def __init__(self, root, num_classes=21, npoints=8192, split='train', datasetname='scannet'):
        self.root = root
        self.num_classes = num_classes
        self.npoints = npoints
        self.split = split
        self.darasetname = datasetname
        self.data_filename = os.path.join(self.root, '%s_%s.pickle' % (datasetname, split))
        try:
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        except Exception as e:
            # print(e)
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp, encoding='latin1')
                self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if split == 'train':
            # labelweights = np.zeros(21)
            labelweights = np.zeros(num_classes)
            for seg in self.semantic_labels_list:
                # tmp, _ = np.histogram(seg, range(22))
                tmp, _ = np.histogram(seg, range(num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test' or split == 'pre':
            # self.labelweights = np.ones(21)
            self.labelweights = np.ones(num_classes)

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        smpmin = np.maximum(coordmax - [1.5, 1.5, 3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax - smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2] - coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :]
            curmin = curcenter - [0.75, 0.75, 1.5]
            curmax = curcenter + [0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(
                vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.scene_points_list)


class DatasetWholeScene():
    def __init__(self, root, num_classes=21, npoints=8192, split='train', datasetname='scannet', step=1.5):
        self.step = step
        self.root = root
        self.num_classes = num_classes
        self.npoints = npoints
        self.split = split
        self.darasetname = datasetname
        self.data_filename = os.path.join(self.root, '%s_%s.pickle' % (datasetname, split))
        # with open(self.data_filename, 'rb') as fp:
        #     self.scene_points_list = pickle.load(fp)
        #     self.semantic_labels_list = pickle.load(fp)
        try:
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        except Exception as e:
            # print(e)
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp, encoding='latin1')
                self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if split == 'train':
            # labelweights = np.zeros(21)
            labelweights = np.zeros(num_classes)
            for seg in self.semantic_labels_list:
                # tmp, _ = np.histogram(seg, range(22))
                tmp, _ = np.histogram(seg, range(num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            # self.labelweights = np.ones(21)
            self.labelweights = np.ones(num_classes)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        # 获取(x,y,z)每一项的最大值，不一定为同一个点
        coordmax = np.max(point_set_ini, axis=0)
        # 获取(x,y,z)每一项的最小值，不一定为同一个点
        coordmin = np.min(point_set_ini, axis=0)
        # nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / self.step).astype(np.int32)
        # nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / self.step).astype(np.int32)
        # tmp print
        # print(nsubvolume_x, nsubvolume_y)
        # 步长自适应
        if nsubvolume_x * nsubvolume_y >= 100:
            self.step = max((coordmax[0] - coordmin[0]) / 10, (coordmax[1] - coordmin[1]) / 10)
            print('STEP change to %s' % self.step)

        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                # curmin = coordmin + [i * 1.5, j * 1.5, 0]
                curmin = coordmin + [i * self.step, j * self.step, 0]
                # curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
                curmax = coordmin + [(i + 1) * self.step, (j + 1) * self.step, coordmax[2] - coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin - 0.2)) * (point_set_ini <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N
                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)


class DatasetPredict():
    def __init__(self, root, num_classes=21, npoints=8192, split='train', datasetname='scannet', step=1.5):
        self.step = step
        self.root = root
        self.num_classes = num_classes
        self.npoints = npoints
        self.split = split
        self.darasetname = datasetname
        self.data_filename = os.path.join(self.root, '%s_%s.pickle' % (datasetname, split))
        # with open(self.data_filename, 'rb') as fp:
        #     self.scene_points_list = pickle.load(fp)
        #     self.semantic_labels_list = pickle.load(fp)
        try:
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        except Exception as e:
            # print(e)
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp, encoding='latin1')
                self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if split == 'train':
            # labelweights = np.zeros(21)
            labelweights = np.zeros(num_classes)
            for seg in self.semantic_labels_list:
                # tmp, _ = np.histogram(seg, range(22))
                tmp, _ = np.histogram(seg, range(num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            # self.labelweights = np.ones(21)
            self.labelweights = np.ones(num_classes)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)

        point_sets = point_set_ini.reshape(1, point_set_ini.shape[0], 3)
        semantic_segs = semantic_seg_ini.reshape(1, semantic_seg_ini.shape[0])
        sample_weights = np.ones(semantic_seg_ini.shape[1]).reshape(1, semantic_seg_ini.shape[1])

        return point_sets, semantic_segs, sample_weights
        # # 获取(x,y,z)每一项的最大值，不一定为同一个点
        # coordmax = np.max(point_set_ini, axis=0)
        # # 获取(x,y,z)每一项的最小值，不一定为同一个点
        # coordmin = np.min(point_set_ini, axis=0)
        # # nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
        # nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / self.step).astype(np.int32)
        # # nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
        # nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / self.step).astype(np.int32)
        # # tmp print
        # # print(nsubvolume_x, nsubvolume_y)
        # # 步长自适应
        # if nsubvolume_x * nsubvolume_y >= 100:
        #     self.step = max((coordmax[0] - coordmin[0]) / 10, (coordmax[1] - coordmin[1]) / 10)
        #     print('STEP change to %s' % self.step)
        #
        # point_sets = list()
        # semantic_segs = list()
        # sample_weights = list()
        # isvalid = False
        # for i in range(nsubvolume_x):
        #     for j in range(nsubvolume_y):
        #         # curmin = coordmin + [i * 1.5, j * 1.5, 0]
        #         curmin = coordmin + [i * self.step, j * self.step, 0]
        #         # curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
        #         curmax = coordmin + [(i + 1) * self.step, (j + 1) * self.step, coordmax[2] - coordmin[2]]
        #         curchoice = np.sum((point_set_ini >= (curmin - 0.2)) * (point_set_ini <= (curmax + 0.2)), axis=1) == 3
        #         cur_point_set = point_set_ini[curchoice, :]
        #         cur_semantic_seg = semantic_seg_ini[curchoice]
        #         if len(cur_semantic_seg) == 0:
        #             continue
        #         mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
        #         choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        #         point_set = cur_point_set[choice, :]  # Nx3
        #         semantic_seg = cur_semantic_seg[choice]  # N
        #         mask = mask[choice]
        #         if sum(mask) / float(len(mask)) < 0.01:
        #             continue
        #         sample_weight = self.labelweights[semantic_seg]
        #         sample_weight *= mask  # N
        #         point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
        #         semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
        #         sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        # point_sets = np.concatenate(tuple(point_sets), axis=0)
        # semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        # sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        # return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)


class DatasetVirtualScan():
    def __init__(self, root, num_classes=21, npoints=8192, split='train', datasetname='scannet'):
        self.root = root
        self.num_classes = num_classes
        self.npoints = npoints
        self.split = split
        self.darasetname = datasetname
        self.data_filename = os.path.join(self.root, '%s_%s.pickle' % (datasetname, split))
        # with open(self.data_filename, 'rb') as fp:
        #     self.scene_points_list = pickle.load(fp)
        #     self.semantic_labels_list = pickle.load(fp)
        try:
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        except Exception as e:
            # print(e)
            with open(self.data_filename, 'rb') as fp:
                self.scene_points_list = pickle.load(fp, encoding='latin1')
                self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if split == 'train':
            # labelweights = np.zeros(21)
            labelweights = np.zeros(num_classes)
            for seg in self.semantic_labels_list:
                # tmp, _ = np.histogram(seg, range(22))
                tmp, _ = np.histogram(seg, range(num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            # self.labelweights = np.ones(21)
            self.labelweights = np.ones(num_classes)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(8):
            smpidx = scene_util.virtual_scan(point_set_ini, mode=i)
            if len(smpidx) < 300:
                continue
            point_set = point_set_ini[smpidx, :]
            semantic_seg = semantic_seg_ini[smpidx]
            sample_weight = sample_weight_ini[smpidx]
            choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
            point_set = point_set[choice, :]  # Nx3
            semantic_seg = semantic_seg[choice]  # N
            sample_weight = sample_weight[choice]  # N
            point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
            sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    # d = ScannetDatasetWholeScene(root='./data', split='test', npoints=8192)
    d = DatasetWholeScene(root='./data', num_classes=21, split='test', npoints=8192, datasetname='scannet')
    labelweights_vox = np.zeros(21)
    for ii in range(len(d)):
        print(ii)
        ps, seg, smpw = d[ii]
        for b in range(ps.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b, smpw[b, :] > 0, :],
                                                                                  seg[b, smpw[b, :] > 0], res=0.02)
            tmp, _ = np.histogram(uvlabel, range(22))
            labelweights_vox += tmp
    print(labelweights_vox[1:].astype(np.float32) / np.sum(labelweights_vox[1:].astype(np.float32)))
    exit()
