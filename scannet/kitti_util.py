"""
Author: Muyangren907
Date: 2019/11/6
"""
import numpy as np
import os
import pprint
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(os.path.join(DATA_DIR, 'training'))
if not os.path.exists(os.path.join(DATA_DIR, 'training')):
    www_list = [
        'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
        'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
        'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
    ]
    for www in www_list:
        zipfile = os.path.basename(www)
        if not os.path.exists(os.path.join(DATA_DIR, zipfile)):
            os.system('wget %s' % www)
            os.system('mv %s %s' % (zipfile, DATA_DIR))
        unzipfile = os.path.join(DATA_DIR, zipfile)
        os.system('unzip %s -d %s' % (unzipfile, DATA_DIR))
        os.system('rm %s' % unzipfile)


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:     这两个rect/ref 是一个东西？
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian  笛卡尔  就是在后面加了一列 1
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''  # n*3 *  3*3  > n*3  > 3*n
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            这部分是我要的，需要把3D坐标转换到激光雷达坐标
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def dealdata2pickle(file_num):
    points_o_list, labelslist = [], []

    save_object_pickle_path = os.path.join(DATA_DIR, 'kitti')

    if not os.path.exists(save_object_pickle_path):
        os.makedirs(save_object_pickle_path)

    lidar_path = os.path.join(DATA_DIR, 'training', 'velodyne')
    label_path = os.path.join(DATA_DIR, 'training', 'label_2')
    calib_path = os.path.join(DATA_DIR, 'training', 'calib')

    for data_id in range(file_num):

        lidar_file = os.path.join(lidar_path, '%06d.bin' % data_id)
        label_file = os.path.join(label_path, '%06d.txt' % data_id)
        calib_file = os.path.join(calib_path, '%06d.txt' % data_id)

        # 加载点云数据
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
        # 仅获取xyz坐标信息，忽略r
        points = points[:, :-1]
        # 获取points的shape
        points_shape = points.shape
        zeros = np.zeros(points_shape[0])
        # 扩充一列，用于记录标签id
        points = np.c_[points, zeros]
        # points[:, 3] = 1
        # print(points)

        # 解析标签文件
        lables = np.loadtxt(label_file, dtype={'names': (
            'type', 'truncated', 'occuluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'h', 'w', 'l', 'x', 'y', 'z',
            'rotation_y'), 'formats': (
            'S14', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
            'float', 'float', 'float')})

        calibs = Calibration(calib_file)

        if lables.size == 1:
            lables = lables[np.newaxis]

        i = 0
        for label in lables:
            i += 1
            # Misc和DontCare
            if label['type'] != b'DontCare':
                # 将图像坐标转换为激光点云坐标
                xyz = calibs.project_rect_to_velo(np.array([[label['x'], label['y'], label['z']]]))

                # 中心点
                x = xyz[0][0]
                y = xyz[0][1]
                z = xyz[0][2]

                # AABB 包围盒，最近点和最远点
                min_point_AABB = [x - label['l'] / 2, y - label['w'] / 2, z, ]
                max_point_AABB = [x + label['l'] / 2, y + label['w'] / 2, z + label['h'], ]

                # 过滤该范围内的激光点
                x_filt = np.logical_and(
                    (points[:, 0] > min_point_AABB[0]), (points[:, 0] < max_point_AABB[0]))
                y_filt = np.logical_and(
                    (points[:, 1] > min_point_AABB[1]), (points[:, 1] < max_point_AABB[1]))
                z_filt = np.logical_and(
                    (points[:, 2] > min_point_AABB[2]), (points[:, 2] < max_point_AABB[2]))
                filt = np.logical_and(x_filt, y_filt)  # 必须同时成立
                filt = np.logical_and(filt, z_filt)  # 必须同时成立

                labelid = 0
                # 标签id定义
                if label['type'] in [b'Pedestrian', b'Person_sitting']:
                    labelid = 1
                elif label['type'] in [b'Car', b'Van', b'Truck', b'Tram']:
                    labelid = 2
                elif label['type'] in [b'Cyclist']:
                    labelid = 3
                elif label['type'] in [b'Misc']:
                    labelid = 4

                points[filt, 3] = labelid

        points_o, labels = points[:, :-1], points[:, -1:].reshape(points_shape[0], )
        points_o_list.append(points_o)
        labelslist.append(labels)
        print('[', data_id + 1, '/', file_num, ']', points_o.shape, labels.shape)

        if (data_id + 1) % 1000 == 0:
            file_name = ''
            if data_id // 1000 == 0:
                file_name = 'kitti_train.pickle'
            else:
                file_name = 'kitti_train_%s.pickle' % (data_id // 1000)
            save_object_pickle_file = os.path.join(save_object_pickle_path, file_name)
            with open(save_object_pickle_file, 'wb') as pf:
                pickle.dump(points_o_list, pf)
                pickle.dump(labelslist, pf)
            print('save', save_object_pickle_file, 'succeed!')
            points_o_list = []
            labelslist = []
            print('clear list succeed!')
        if data_id + 1 == file_num:
            save_object_pickle_file = os.path.join(save_object_pickle_path, 'kitti_test.pickle')
            with open(save_object_pickle_file, 'wb') as pf:
                pickle.dump(points_o_list, pf)
                pickle.dump(labelslist, pf)
            print('save', save_object_pickle_file, 'succeed!')
            points_o_list = []
            labelslist = []
            print('clear list succeed!')


# 获取文件夹下文件个数
def getfilenum(path):
    return len(os.listdir(path))


if __name__ == '__main__':
    # typelist = {b'Pedestrian': 4487, b'Truck': 1094, b'Car': 28742, b'Cyclist': 1627, b'DontCare': 11295, b'Misc': 973,
    #             b'Van': 2914, b'Tram': 511, b'Person_sitting': 222}
    file_num_path = os.path.join(DATA_DIR, 'training', 'velodyne')
    file_num = getfilenum(file_num_path)
    dealdata2pickle(file_num)
