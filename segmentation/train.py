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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
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
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', type=str, default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
# add data directory parser
parser.add_argument('--dataset', type=str, default='scannet',
                    help='Dataset name [default: scannet]')
parser.add_argument('--downloader', type=str, default='wget', help='Downloader for download dataset')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
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

# NUM_CLASSES = 21
# NUM_CLASSES = 5

# Shapenet official train/test split
# DATA_PATH = os.path.join(ROOT_DIR, 'data', 'scannet_data_pointnet2')
DATA_PATH = os.path.join(ROOT_DIR, 'data', DATASET)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

NUM_CLASSES, STEP = dataset_util.deal_dataset(DATASET, DOWNLOADER, DATA_PATH)

# TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train')
# TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test')
# TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')

# root, num_classes=21, npoints=8192, split='train', datasetname='scannet'


TRAIN_DATASET = dataset.Dataset(root=DATA_PATH, num_classes=NUM_CLASSES, npoints=NUM_POINT, split='train',
                                datasetname=DATASET)
TEST_DATASET = dataset.Dataset(root=DATA_PATH, num_classes=NUM_CLASSES, npoints=NUM_POINT, split='test',
                               datasetname=DATASET)
TEST_DATASET_WHOLE_SCENE = dataset.DatasetWholeScene(root=DATA_PATH, num_classes=NUM_CLASSES, npoints=NUM_POINT,
                                                     split='test', datasetname=DATASET, step=STEP)


# if DATASET == 'scannet':
#     NUM_CLASSES = 21
#     www = 'https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip'
#     zipfile = os.path.basename(www)
#     if not os.path.exists(os.path.join(DATA_PATH, 'scannet_data_pointnet2.zip')):
#         if DOWNLOADER == 'wget':
#             os.system('wget %s' % www)
#         elif DOWNLOADER == 'aria2':
#             os.system('aria2c -x 15 -s 15 %s' % www)
#         os.system('mv %s %s' % (zipfile, DATA_PATH))
#     if not os.path.exists(os.path.join(DATA_PATH, 'scannet_train.pickle')):
#         unzipfile = os.path.join(DATA_PATH, zipfile)
#         os.system('unzip -q %s -d %s' % (unzipfile, DATA_PATH))
#         os.system('mv %s/* %s' % (os.path.join(DATA_PATH, 'data'), DATA_PATH))
#         os.system('rmdir %s' % (os.path.join(DATA_PATH, 'data')))
#         # os.system('rm %s' % unzipfile)
#         # path_old = os.path.join(ROOT_DIR, 'data', 'data')
#         # path_new = os.path.join(ROOT_DIR, 'data', 'scannet_data_pointnet2')
#         # os.system('mv %s %s' % (path_old, path_new))
#     TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train')
#     TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test')
#     TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')


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


def train():
    global EPOCH_CNT
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
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

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

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
        if os.path.exists(best_model_file):
            save_path = saver.restore(sess, best_model_file)
            log_string("Model load from file: %s" % save_path)

        for epoch in range(MAX_EPOCH):
            log_string('\n**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            if epoch % 5 == 0:
                EPOCH_CNT = epoch
                acc = eval_one_epoch(sess, ops, test_writer)
                acc = eval_whole_scene_one_epoch(sess, ops, test_writer)
            if acc > best_acc:
                best_acc = acc
                # save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % (epoch)))
                save_path = saver.save(sess, best_model_file)
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, model_file)
                log_string("Model saved in file: %s" % save_path)


def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw

        dropout_ratio = np.random.random() * 0.875  # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
        batch_data[i, drop_idx, :] = batch_data[i, 0, :]
        batch_label[i, drop_idx] = batch_label[i, 0]
        batch_smpw[i, drop_idx] *= 0
    return batch_data, batch_label, batch_smpw


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


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
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
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        if (batch_idx + 1) % 10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0


# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    # global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET) // BATCH_SIZE
    # print('num_batches', num_batches)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('\n---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    # labelweights = np.zeros(21)
    # labelweights_vox = np.zeros(21)
    labelweights = np.zeros(NUM_CLASSES)
    labelweights_vox = np.zeros(NUM_CLASSES)
    for batch_idx in range(num_batches):
        print('[', batch_idx + 1, '/', num_batches, ']', end='\r')
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        aug_data = provider.rotate_point_cloud_z(batch_data)

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        # tmp print
        # print(batch_idx)
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)  # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (
                batch_smpw > 0))  # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
        loss_sum += loss_val
        # tmp, _ = np.histogram(batch_label, range(22))
        tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))

        for b in range(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b, batch_smpw[b, :] > 0, :],
                                                                                  np.concatenate((np.expand_dims(
                                                                                      batch_label[
                                                                                          b, batch_smpw[b, :] > 0], 1),
                                                                                                  np.expand_dims(
                                                                                                      pred_val[
                                                                                                          b, batch_smpw[
                                                                                                             b, :] > 0],
                                                                                                      1)), axis=1),
                                                                                  res=0.02)
            total_correct_vox += np.sum((uvlabel[:, 0] == uvlabel[:, 1]) & (uvlabel[:, 0] > 0))
            total_seen_vox += np.sum(uvlabel[:, 0] > 0)
            # tmp, _ = np.histogram(uvlabel[:, 0], range(22))
            tmp, _ = np.histogram(uvlabel[:, 0], range(NUM_CLASSES + 1))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
                total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point accuracy vox: %f' % (total_correct_vox / float(total_seen_vox)))
    log_string('eval point avg class acc vox: %f' % (
        np.mean(np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6))))
    log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (
        np.mean(np.array(total_correct_class[1:]) / (np.array(total_seen_class[1:], dtype=np.float) + 1e-6))))
    labelweights_vox = labelweights_vox[1:].astype(np.float32) / np.sum(labelweights_vox[1:].astype(np.float32))
    # caliweights = np.array(
    #     [0.388, 0.357, 0.038, 0.033, 0.017, 0.02, 0.016, 0.025, 0.002, 0.002, 0.002, 0.007, 0.006, 0.022, 0.004, 0.0004,
    #      0.003, 0.002, 0.024, 0.029])
    # log_string('eval point calibrated average acc: %f' % (
    #     np.average(np.array(total_correct_class[1:]) / (np.array(total_seen_class[1:], dtype=np.float) + 1e-6),
    #                weights=caliweights)))
    per_class_str = 'vox based --------\n'
    for l in range(1, NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f \n' % (
            l, labelweights_vox[l - 1], total_correct_class[l] / (float(total_seen_class[l]) + 1e-6))
    log_string(per_class_str)
    # EPOCH_CNT += 1
    return total_correct / float(total_seen)


# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    # global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE_SCENE))
    num_batches = len(TEST_DATASET_WHOLE_SCENE)
    # print('num_batches', num_batches)
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('\n---- EPOCH %03d EVALUATION WHOLE SCENE----' % (EPOCH_CNT))

    # labelweights = np.zeros(21)
    # labelweights_vox = np.zeros(21)
    labelweights = np.zeros(NUM_CLASSES)
    labelweights_vox = np.zeros(NUM_CLASSES)
    is_continue_batch = False

    extra_batch_data = np.zeros((0, NUM_POINT, 3))
    extra_batch_label = np.zeros((0, NUM_POINT))
    extra_batch_smpw = np.zeros((0, NUM_POINT))
    for batch_idx in range(num_batches):
        # tmp print
        print('[', batch_idx + 1, '/', num_batches, ']', end='\r')
        # 数据加载时，若分割不对，即步长太小会导致内存占用过大而killed
        # 步长详见 dataset.py DatasetWholeScene 中 step
        if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
            batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
            batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
        else:
            batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
            batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
            batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)
        if batch_data.shape[0] < BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0] == BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0, NUM_POINT, 3))
            extra_batch_label = np.zeros((0, NUM_POINT))
            extra_batch_smpw = np.zeros((0, NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:, :, :]
            extra_batch_label = batch_label[BATCH_SIZE:, :]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:, :]
            batch_data = batch_data[:BATCH_SIZE, :, :]
            batch_label = batch_label[:BATCH_SIZE, :]
            batch_smpw = batch_smpw[:BATCH_SIZE, :]

        # print('load ok')

        aug_data = batch_data
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)  # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (
                batch_smpw > 0))  # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
        loss_sum += loss_val
        # tmp, _ = np.histogram(batch_label, range(22))
        tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))

        for b in range(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b, batch_smpw[b, :] > 0, :],
                                                                                  np.concatenate((np.expand_dims(
                                                                                      batch_label[
                                                                                          b, batch_smpw[b, :] > 0], 1),
                                                                                                  np.expand_dims(
                                                                                                      pred_val[
                                                                                                          b, batch_smpw[
                                                                                                             b, :] > 0],
                                                                                                      1)), axis=1),
                                                                                  res=0.02)
            total_correct_vox += np.sum((uvlabel[:, 0] == uvlabel[:, 1]) & (uvlabel[:, 0] > 0))
            total_seen_vox += np.sum(uvlabel[:, 0] > 0)
            # tmp, _ = np.histogram(uvlabel[:, 0], range(22))
            tmp, _ = np.histogram(uvlabel[:, 0], range(NUM_CLASSES + 1))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
                total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))
        # tmp print
        # print(batch_idx, 'ok')
    log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval whole scene point accuracy vox: %f' % (total_correct_vox / float(total_seen_vox)))
    log_string('eval whole scene point avg class acc vox: %f' % (
        np.mean(np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6))))
    log_string('eval whole scene point accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval whole scene point avg class acc: %f' % (
        np.mean(np.array(total_correct_class[1:]) / (np.array(total_seen_class[1:], dtype=np.float) + 1e-6))))
    labelweights = labelweights[1:].astype(np.float32) / np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32) / np.sum(labelweights_vox[1:].astype(np.float32))
    # caliweights = np.array(
    #     [0.388, 0.357, 0.038, 0.033, 0.017, 0.02, 0.016, 0.025, 0.002, 0.002, 0.002, 0.007, 0.006, 0.022, 0.004, 0.0004,
    #      0.003, 0.002, 0.024, 0.029])
    # caliacc = np.average(
    #     np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6),
    #     weights=caliweights)
    # log_string('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------\n'
    for l in range(1, NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f\n' % (
            l, labelweights_vox[l - 1], total_correct_class_vox[l] / (float(total_seen_class_vox[l]) + 1e-6))
    log_string(per_class_str)
    # EPOCH_CNT += 1
    # return caliacc
    return total_correct / float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
