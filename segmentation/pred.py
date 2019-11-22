import argparse
import math
from datetime import datetime
# import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(BASE_DIR)  # model
sys.path.append(ROOT_DIR)  # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider
import tf_util
import pc_util

import dataset_util

sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import dataset
import scannet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--model_load', type=str, default='kitti.ckpt', help='Model data file name')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
# parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 1]')
# parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', type=str, default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
# add data directory parser
parser.add_argument('--dataset', type=str, default='kitti',
                    help='Dataset name [default: kitti]')
parser.add_argument('--downloader', type=str, default='wget', help='Downloader for download dataset')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

# BATCH_SIZE = FLAGS.batch_size
BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
# MAX_EPOCH = FLAGS.max_epoch
MAX_EPOCH = 1
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
# add data directory parser
DATASET = FLAGS.dataset
DOWNLOADER = FLAGS.downloader

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

DATA_PATH = os.path.join(ROOT_DIR, 'data', DATASET)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

NUM_CLASSES, STEP = dataset_util.deal_dataset(DATASET, DOWNLOADER, DATA_PATH)

PREDICT_DATASET = dataset.DatasetPredict(root=DATA_PATH, num_classes=NUM_CLASSES, npoints=NUM_POINT, split='pre',
                                         datasetname=DATASET)


# if DATASET == 'kitti':
#     DATA_PRE_DIR = os.path.join(DATA_PATH, 'testing', 'velodyne')
#     data_pre_list = os.listdir(DATA_PRE_DIR)
#     data_num = len(data_pre_list)

# exit(0)
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def predict():
    global EPOCH_CNT
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            # pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, None)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss ---")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            # correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * -1)
            # tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator ---")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        model_save_path = os.path.join(LOG_DIR, "model_save")

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        best_model_file = os.path.join(model_save_path, "%s_best.ckpt" % DATASET)
        model_file = os.path.join(model_save_path, "%s.ckpt" % DATASET)
        # Load model
        # print(best_model_file)
        # print(os.path.exists(best_model_file + '.index'))
        if os.path.exists(best_model_file + '.index'):
            saver.restore(sess, best_model_file)
            log_string("Model load from file: %s" % best_model_file)

        save_object_pickle_path = os.path.join(DATA_PATH, 'predict')
        if not os.path.exists(save_object_pickle_path):
            os.makedirs(save_object_pickle_path)

        for epoch in range(MAX_EPOCH):
            log_string('\n**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            point_epoch_list, labels_epoch_list = predict_one_epoch(sess, ops, train_writer)
            save_object_pickle_file = os.path.join(save_object_pickle_path, 'kitti_predict_%03d.pickle' % epoch)
            with open(save_object_pickle_file, 'wb') as pf:
                pickle.dump(point_epoch_list, pf)
                pickle.dump(labels_epoch_list, pf)
            print('save', save_object_pickle_file, 'succeed!')


# def get_batch_wdp(dataset, idxs, start_idx, end_idx):
#     # bsize = end_idx - start_idx
#     # batch_data = np.zeros((bsize, NUM_POINT, 3))
#     # batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
#     # batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
#     # for i in range(bsize):
#     #     ps, seg, smpw = dataset[idxs[i + start_idx]]
#     #     batch_data[i, ...] = ps
#     #     batch_label[i, :] = seg
#     #     batch_smpw[i, :] = smpw
#     #
#     #     dropout_ratio = np.random.random() * 0.875  # 0-0.875
#     #     drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
#     #     batch_data[i, drop_idx, :] = batch_data[i, 0, :]
#     #     batch_label[i, drop_idx] = batch_label[i, 0]
#     #     batch_smpw[i, drop_idx] *= 0
#     # return batch_data, batch_label, batch_smpw
#     batch_data,batch_label = dataset


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw
    return batch_data, batch_label, batch_smpw


def predict_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    # train_idxs = np.arange(0, len(TRAIN_DATASET))
    # 预测不需要打乱
    # np.random.shuffle(train_idxs)
    # num_batches = len(TRAIN_DATASET) // BATCH_SIZE
    num_batches = len(PREDICT_DATASET)

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    point_epoch_list = []
    labels_epoch_list = []
    for batch_idx in range(num_batches):
        print('[ %03d / %03d ]' % (batch_idx + 1, num_batches), end='\r')
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        # batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        batch_data, batch_label, batch_smpw = PREDICT_DATASET[batch_idx]

        # batch_data = batch_data.reshape(1, batch_data.shape[0], 3)
        # batch_label = batch_label.reshape(1, batch_label.shape[0])
        # batch_smpw = np.ones(batch_label.shape[1]).reshape(1, batch_label.shape[1])

        # Augment batched point clouds by rotation
        aug_data = provider.rotate_point_cloud_z(batch_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)

        # 数据放入list
        point_epoch_list.append(batch_data.reshape(batch_data.shape[1], batch_data.shape[2]))
        labels_epoch_list.append(pred_val.reshape(pred_val.shape[1]))
        # correct = np.sum(pred_val == batch_label)
        # total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        # loss_sum += loss_val
        # if (batch_idx + 1) % 10 == 0:
        #     log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
        #     log_string('mean loss: %f' % (loss_sum / 10))
        #     log_string('accuracy: %f' % (total_correct / float(total_seen)))
        #     total_correct = 0
        #     total_seen = 0
        #     loss_sum = 0

    return point_epoch_list, labels_epoch_list


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    predict()
    LOG_FOUT.close()
