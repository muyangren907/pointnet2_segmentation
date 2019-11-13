import os
import kitti_util


def deal_dataset(DATASET, DOWNLOADER, DATA_PATH):
    NUM_CLASSES = 21
    if DATASET == 'scannet':
        NUM_CLASSES = 21
        www = 'https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip'
        zipfile = os.path.basename(www)
        train_file = os.path.join(DATA_PATH, 'scannet_train.pickle')
        test_file = os.path.join(DATA_PATH, 'scannet_test.pickle')
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            if not os.path.exists(os.path.join(DATA_PATH, zipfile)):
                cmd = 'wget %s' % www
                if DOWNLOADER == 'aria2':
                    # os.system('aria2c -x 15 -s 15 %s' % www)
                    cmd = 'aria2c -x 15 -s 15 %s' % www
                elif DOWNLOADER == 'axel':
                    cmd = 'axel -n 64 %s' % www
                print(cmd)
                os.system(cmd)
                cmd = 'mv %s %s' % (zipfile, DATA_PATH)
                # print(cmd)
                os.system(cmd)
            # if not os.path.exists(os.path.join(DATA_PATH, 'scannet_train.pickle')):
            unzipfile = os.path.join(DATA_PATH, zipfile)
            # os.system('unzip -q %s -d %s' % (unzipfile, DATA_PATH))
            cmd = 'unzip -q %s -d %s' % (unzipfile, DATA_PATH)
            print(cmd)
            os.system(cmd)
            os.system('mv %s/* %s' % (os.path.join(DATA_PATH, 'data'), DATA_PATH))
            os.system('rmdir %s' % (os.path.join(DATA_PATH, 'data')))
    elif DATASET == 'kitti':
        NUM_CLASSES = 9
        train_file = os.path.join(DATA_PATH, 'kitti_train.pickle')
        test_file = os.path.join(DATA_PATH, 'kitti_test.pickle')
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            www_list = [
                'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
                'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
                'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
            ]
            for www in www_list:
                zipfile = os.path.basename(www)
                if not os.path.exists(os.path.join(DATA_PATH, zipfile)):
                    cmd = 'wget %s' % www
                    if DOWNLOADER == 'aria2':
                        # os.system('aria2c -x 15 -s 15 %s' % www)
                        cmd = 'aria2c -x 15 -s 15 %s' % www
                    elif DOWNLOADER == 'axel':
                        cmd = 'axel -n 64 %s' % www
                    print(cmd)
                    os.system(cmd)
                    os.system('mv %s %s' % (zipfile, DATA_PATH))
                    unzipfile = os.path.join(DATA_PATH, zipfile)
                    cmd = 'unzip -q %s -d %s' % (unzipfile, DATA_PATH)
                    print(cmd)
                    os.system(cmd)
            # file_num_path = os.path.join(DATA_PATH, 'training', 'velodyne')
            # file_num = len(os.listdir(file_num_path))
            kitti_util.main()
            # for i in range(0, file_num, 1000):
            #     # gc.collect()
            #     if i + 1000 < file_num:
            #         kitti_util.dealdata2pickle(i, i + 1000,file_num)
            #     else:
            #         kitti_util.dealdata2pickle(i, file_num,file_num)
    print('*' * 6)
    print('DATASET:', DATASET, '\nDATA_PATH:', DATA_PATH, '\nNUM_CLASSES:', NUM_CLASSES)
    print('*' * 6)
    return NUM_CLASSES


if __name__ == '__main__':
    pass
