### ScanNet Data

Original dataset website: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

You can get our preprocessed data at <a href="https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip">here (1.72GB)</a> and refer to the code in `scannet_util.py` for data loading. Note that the virtual scan data is generated on the fly from our preprocessed data.

Some code we used for scannet preprocessing is also included in `preprocessing` folder. You have to download the original ScanNet data and make small modifications in paths in order to run them.

Note: To use ScanNetV2 data, change the tsv file to `scannetv2-labels.combined.tsv` and also update `scannet_util.py` to read the raw class and NYU40 names in the right columns (shifted by 1 compared to the V1 tsv).

### KITTI Data

Make sure your hard drive has at least 60GB of free space

- Velodyne point clouds (29 GB) 

    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

- Training labels of object data set (5 MB)

    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

- Camera calibration matrices of object data set (16 MB)

    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

`kitti_utill.py` will help you download and unzip them, so you should run it before you run train.py