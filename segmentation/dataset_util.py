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
                if DOWNLOADER == 'wget':
                    os.system('wget %s' % www)
                elif DOWNLOADER == 'aria2':
                    os.system('aria2c -x 15 -s 15 %s' % www)
                os.system('mv %s %s' % (zipfile, DATA_PATH))
        # if not os.path.exists(os.path.join(DATA_PATH, 'scannet_train.pickle')):
            unzipfile = os.path.join(DATA_PATH, zipfile)
            os.system('unzip -q %s -d %s' % (unzipfile, DATA_PATH))
            os.system('mv %s/* %s' % (os.path.join(DATA_PATH, 'data'), DATA_PATH))
            os.system('rmdir %s' % (os.path.join(DATA_PATH, 'data')))
    elif DATASET == 'kitti':
        NUM_CLASSES = 5

        www_list = [
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
        ]
        for www in www_list:
            zipfile = os.path.basename(www)
            if not os.path.exists(os.path.join(DATA_PATH, zipfile)):
                if DOWNLOADER == 'wget':
                    os.system('wget %s' % www)
                elif DOWNLOADER == 'aria2':
                    os.system('aria2c -x 15 -s 15 %s' % www)
                os.system('mv %s %s' % (zipfile, DATA_PATH))
                unzipfile = os.path.join(DATA_PATH, zipfile)
                os.system('unzip -q %s -d %s' % (unzipfile, DATA_PATH))

        kitti_util.main()

    return NUM_CLASSES


if __name__ == '__main__':
    pass
